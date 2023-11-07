import click
from train import LitForcedAlignmentModel
import pathlib
import torch
# import textgrid
from modules.utils.load_wav import load_wav
from modules.utils.get_melspec import MelSpecExtractor
import numpy as np
from modules.utils.plot import plot_for_test
import lightning as pl


class ForcedAlignmentModelInferer:
    def __init__(self, ckpt_path: str, dictionary: str, device):
        self.model = LitForcedAlignmentModel.load_from_checkpoint(ckpt_path)
        self.model.eval()

        with open(dictionary, 'r') as f:
            dictionary = f.read().strip().split('\n')
        self.dictionary = {item.split('\t')[0].strip(): item.split('\t')[1].strip().split(' ')
                           for item in dictionary}

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model.to(self.device)

        self.sample_rate = self.model.hparams.melspec_config["sample_rate"]
        self.get_melspec = MelSpecExtractor(**self.model.hparams.melspec_config, device=self.device)

    def infer(self, input_folder: str, output_folder: str):
        # load dataset list
        wav_path_list = pathlib.Path(input_folder).rglob('*.wav')
        for wav_path in wav_path_list:
            ph_seq_pred, ph_dur_pred, ctc_pred, _ = self.infer_once(wav_path)
            if ph_seq_pred is None:
                continue
            # self.get_textgrid()

    def get_textgrid(self, ph_seq, ph_dur):
        pass

    def infer_once(self, wav_path, return_ctc=False, return_plot=False):

        lab_path = wav_path.parent / f"{wav_path.stem}.lab"
        if not lab_path.exists():
            return None, None, None, None

        # TODO: add phoneme mode, add matching mode
        with open(lab_path, 'r') as f:
            word_seq = f.read().strip().split(' ')
        ph_seq = [0]
        for idx, word in enumerate(word_seq):
            ph_seq.extend(self.dictionary[word])
            ph_seq.append(0)
        ph_seq_id = np.array([self.model.vocab[ph] if ph != 0 else 0 for ph in ph_seq])

        # forward
        waveform = load_wav(wav_path, self.device, self.sample_rate)
        melspec = self.get_melspec(waveform).unsqueeze(0)
        melspec = (melspec - melspec.mean()) / melspec.std()
        with torch.no_grad():
            (
                ph_frame_pred,  # (B, T, vocab_size)
                ph_edge_pred,  # (B, T, 2)
                ctc_pred,  # (B, T, vocab_size)
            ) = self.model(melspec.transpose(1, 2))

        ph_frame_pred = ph_frame_pred.squeeze(0)
        ph_edge_pred = ph_edge_pred.squeeze(0)
        ctc_pred = ctc_pred.squeeze(0)

        # decode
        edge = torch.softmax(ph_edge_pred, dim=-1).cpu().numpy().astype("float64")
        edge_diff = np.pad(np.diff(edge[:, 0]), (1, 0), "constant", constant_values=0)
        edge_prob = np.pad(edge[1:, 0] + edge[:-1, 0], (1, 0), "constant", constant_values=edge[0, 0] * 2).clip(0, 1)

        ph_prob_log = torch.log_softmax(ph_frame_pred, dim=-1).cpu().numpy().astype("float64")
        (
            ph_seq_id_pred,
            ph_time_int_pred,
            frame_confidence,
        ) = self.decode(
            ph_seq_id,
            ph_prob_log,
            edge_prob,
        )

        # postprocess
        ph_seq_pred = np.array([self.model.vocab[ph] for ph in ph_seq_id_pred])
        ph_time_fractional = (edge_diff[ph_time_int_pred] / 2).clip(-0.5, 0.5)
        ph_time_pred = ph_time_int_pred.astype("float64") + ph_time_fractional
        ph_time_pred = ph_time_pred * (self.model.hparams.melspec_config["hop_length"] / self.sample_rate)
        ph_time_pred = np.concatenate([ph_time_pred, [ph_frame_pred.shape[0]]])
        ph_dur_pred = np.diff(ph_time_pred)

        # ctc decode
        ctc = None
        if return_ctc:
            ctc = torch.argmax(ctc_pred, dim=-1).cpu().numpy()
            ctc = np.unique(ctc)
            ctc = np.array([self.model.vocab[ph] for ph in ctc if ph != 0])

        # plot
        fig = None
        if return_plot:
            ph_frame_idx = np.zeros(ph_frame_pred.shape[0], dtype="int32")
            ph_frame_idx[ph_time_int_pred] = 1
            ph_frame_idx = ph_frame_idx.cumsum() - 1
            ph_frame_id_gt = ph_seq_id_pred[ph_frame_idx]
            raw = {
                "melspec": melspec.cpu().numpy(),
                "ph_seq": ph_seq_pred,
                "ph_time": ph_time_int_pred.astype("float64") + ph_time_fractional,
                "frame_confidence": frame_confidence,

                "ph_frame_prob": torch.softmax(ph_frame_pred, dim=-1).cpu().numpy(),
                "ph_frame_id_gt": ph_frame_id_gt,
                "edge_prob": edge_prob,
            }
            fig = plot_for_test(**raw)

        return ph_seq_pred, ph_dur_pred, ctc, fig

    def decode(self, ph_seq_id, ph_prob_log, edge_prob):
        # ph_seq_id: (T)
        # ph_prob_log: (T, vocab_size)
        # edge_prob: (T,2)
        T = ph_prob_log.shape[0]
        S = len(ph_seq_id)

        edge_prob_log = np.log(edge_prob).astype("float64")
        not_edge_prob_log = np.log(1 - edge_prob).astype("float64")
        # 乘上is_phoneme正确分类的概率 TODO: enable this
        # ph_prob_log[:, 0] += ph_prob_log[:, 0]
        # ph_prob_log[:, 1:] += 1 / ph_prob_log[:, [0]]

        # init
        dp = np.zeros([T, S]).astype("float64") - np.inf  # (T, S)
        backtrack_s = np.zeros_like(dp).astype("int32") - 1
        # 只能从<EMPTY>开始或者从第一个音素开始
        dp[0, 0] = ph_prob_log[0, 0]
        dp[0, 1] = ph_prob_log[0, ph_seq_id[1]]
        # forward
        for t in range(1, T):
            # [t-1,s] -> [t,s]
            prob1 = dp[t - 1, :] + ph_prob_log[t, ph_seq_id[:]] + not_edge_prob_log[t]
            # [t-1,s-1] -> [t,s]
            prob2 = dp[t - 1, :-1] + ph_prob_log[t, ph_seq_id[:-1]] + edge_prob_log[t]
            prob2 = np.pad(prob2, (1, 0), "constant", constant_values=-np.inf)
            # [t-1,s-2] -> [t,s]
            prob3 = dp[t - 1, :-2] + ph_prob_log[t, ph_seq_id[:-2]] + edge_prob_log[t]
            prob3[ph_seq_id[1:-1] != 0] = -np.inf  # 不能跳过音素，可以跳过<EMPTY>
            prob3 = np.pad(prob3, (2, 0), "constant", constant_values=-np.inf)

            backtrack_s[t, :] = np.argmax(np.stack([prob1, prob2, prob3]), axis=0)
            dp[t, :] = np.max(np.stack([prob1, prob2, prob3]), axis=0)

        # backward
        ph_seq_id_pred = []
        ph_time_int = []
        frame_confidence = []
        # 只能从最后一个音素或者<EMPTY>结束
        if dp[-1, -2] > dp[-1, -1]:
            s = S - 2
        else:
            s = S - 1
        for t in np.arange(T - 1, -1, -1):
            assert backtrack_s[t, s] >= 0 or t == 0
            frame_confidence.append(dp[t, s])
            if backtrack_s[t, s] != 0:
                ph_seq_id_pred.append(ph_seq_id[s])
                ph_time_int.append(t)
                s -= backtrack_s[t, s]
        ph_seq_id_pred.reverse()
        ph_time_int.reverse()
        frame_confidence.reverse()
        frame_confidence = np.exp(np.diff(np.pad(frame_confidence, (1, 0), 'constant', constant_values=0.), 1))

        return (
            np.array(ph_seq_id_pred),
            np.array(ph_time_int),
            np.array(frame_confidence),
        )


@click.command()
@click.option('--ckpt', '-c',
              default='ckpt/mandarin_opencpop-extension_singing/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt',
              type=str, help='path to the checkpoint')
@click.option('--input', '-i', default='segments', type=str, help='path to the input folder')
@click.option('--output', '-o', default='segments', type=str, help='path to the output folder')
@click.option('--dictionary', '-d', default='dictionary/opencpop-extension.txt', type=str,
              help='path to the dictionary')
@click.option('--phoneme', '-p', default=False, is_flag=True, help='use phoneme mode')
@click.option("--matching", "-m", default=False, is_flag=True, help="use lyric matching mode")
@click.option('--device', '-d', default=None, type=str, help='device to use')
def main(ckpt, input, output, **kwargs):  # dictionary, phoneme, matching, device
    torch.set_grad_enabled(False)
    model = LitForcedAlignmentModel.load_from_checkpoint(ckpt)
    model.set_infer_params(kwargs)
    print(model.infer_params)
    # wav_path_list = pathlib.Path(input).rglob('*.wav')
    # trainer = pl.Trainer()
    # predictions = trainer.predict(model, dataloaders=wav_path_list, return_predictions=True)
    # save_textgrids(predictions, output)
    # save_htk(predictions, output)
    # save_transcriptions(predictions, output)


if __name__ == "__main__":
    main()
