import os
import pathlib

import click
import numpy as np
import onnxruntime as ort
import torchaudio
import yaml
from tqdm import tqdm

import modules.AP_detector
import modules.g2p
import numba

from modules.utils.export_tool import Exporter
from modules.utils.post_processing import post_processing


def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def run_inference(session, waveform, num_frames, ph_seq_id):
    output_names = [output.name for output in session.get_outputs()]

    input_data = {
        'waveform': waveform,
        'num_frames': np.array(num_frames, dtype=np.int64),
        'ph_seq_id': ph_seq_id
    }

    # 运行推理
    try:
        results = session.run(output_names, input_data)
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        raise

    # 将结果转换为字典形式
    output_dict = {name: result for name, result in zip(output_names, results)}

    return output_dict


def create_session(onnx_model_path):
    providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'
                 ]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    try:
        session = ort.InferenceSession(onnx_model_path, sess_options=session_options, providers=providers)
    except Exception as e:
        print(f"An error occurred while creating ONNX Runtime session: {e}")
        raise

    return session


@numba.jit
def forward_pass(T, S, prob_log, not_edge_prob_log, edge_prob_log, curr_ph_max_prob_log, dp, backtrack_s, ph_seq_id,
                 prob3_pad_len):
    for t in range(1, T):
        # [t-1,s] -> [t,s]
        prob1 = dp[t - 1, :] + prob_log[t, :] + not_edge_prob_log[t]

        prob2 = np.empty(S, dtype=np.float32)
        prob2[0] = -np.inf
        for i in range(1, S):
            prob2[i] = (
                    dp[t - 1, i - 1]
                    + prob_log[t, i - 1]
                    + edge_prob_log[t]
                    + curr_ph_max_prob_log[i - 1] * (T / S)
            )

        # [t-1,s-2] -> [t,s]
        prob3 = np.empty(S, dtype=np.float32)
        for i in range(prob3_pad_len):
            prob3[i] = -np.inf
        for i in range(prob3_pad_len, S):
            if i - prob3_pad_len + 1 < S - 1 and ph_seq_id[i - prob3_pad_len + 1] != 0:
                prob3[i] = -np.inf
            else:
                prob3[i] = (
                        dp[t - 1, i - prob3_pad_len]
                        + prob_log[t, i - prob3_pad_len]
                        + edge_prob_log[t]
                        + curr_ph_max_prob_log[i - prob3_pad_len] * (T / S)
                )

        stacked_probs = np.empty((3, S), dtype=np.float32)
        for i in range(S):
            stacked_probs[0, i] = prob1[i]
            stacked_probs[1, i] = prob2[i]
            stacked_probs[2, i] = prob3[i]

        for i in range(S):
            max_idx = 0
            max_val = stacked_probs[0, i]
            for j in range(1, 3):
                if stacked_probs[j, i] > max_val:
                    max_val = stacked_probs[j, i]
                    max_idx = j
            dp[t, i] = max_val
            backtrack_s[t, i] = max_idx

        for i in range(S):
            if backtrack_s[t, i] == 0:
                curr_ph_max_prob_log[i] = max(curr_ph_max_prob_log[i], prob_log[t, i])
            elif backtrack_s[t, i] > 0:
                curr_ph_max_prob_log[i] = prob_log[t, i]

        for i in range(S):
            if ph_seq_id[i] == 0:
                curr_ph_max_prob_log[i] = 0

    return dp, backtrack_s, curr_ph_max_prob_log


def decode(ph_seq_id, ph_prob_log, edge_prob):
    # ph_seq_id: (S)
    # ph_prob_log: (T, vocab_size)
    # edge_prob: (T,2)
    T = ph_prob_log.shape[0]
    S = len(ph_seq_id)
    # not_SP_num = (ph_seq_id > 0).sum()
    prob_log = ph_prob_log[:, ph_seq_id]

    edge_prob_log = np.log(edge_prob + 1e-6).astype("float32")
    not_edge_prob_log = np.log(1 - edge_prob + 1e-6).astype("float32")

    # init
    curr_ph_max_prob_log = np.full(S, -np.inf)
    dp = np.full((T, S), -np.inf, dtype="float32")  # (T, S)
    backtrack_s = np.full_like(dp, -1, dtype="int32")

    dp[0, 0] = prob_log[0, 0]
    curr_ph_max_prob_log[0] = prob_log[0, 0]
    if ph_seq_id[0] == 0 and prob_log.shape[-1] > 1:
        dp[0, 1] = prob_log[0, 1]
        curr_ph_max_prob_log[1] = prob_log[0, 1]

    # forward
    prob3_pad_len = 2 if S >= 2 else 1
    dp, backtrack_s, curr_ph_max_prob_log = forward_pass(
        T, S, prob_log, not_edge_prob_log, edge_prob_log, curr_ph_max_prob_log, dp, backtrack_s, ph_seq_id,
        prob3_pad_len
    )

    # backward
    ph_idx_seq = []
    ph_time_int = []
    frame_confidence = []

    # 如果mode==forced，只能从最后一个音素或者SP结束
    if S >= 2 and dp[-1, -2] > dp[-1, -1] and ph_seq_id[-1] == 0:
        s = S - 2
    else:
        s = S - 1

    for t in np.arange(T - 1, -1, -1):
        assert backtrack_s[t, s] >= 0 or t == 0
        frame_confidence.append(dp[t, s])
        if backtrack_s[t, s] != 0:
            ph_idx_seq.append(s)
            ph_time_int.append(t)
            s -= backtrack_s[t, s]
    ph_idx_seq.reverse()
    ph_time_int.reverse()
    frame_confidence.reverse()
    frame_confidence = np.exp(
        np.diff(
            np.pad(frame_confidence, (1, 0), "constant", constant_values=0.0), 1
        )
    )

    return (
        np.array(ph_idx_seq),
        np.array(ph_time_int),
        np.array(frame_confidence),
    )


@click.command()
@click.option(
    "--onnx",
    "-c",
    default=None,
    required=True,
    type=str,
    help="path to the onnx",
)
@click.option(
    "--folder", "-f", default="segments", type=str, help="path to the input folder"
)
@click.option(
    "--g2p", "-g", default="Dictionary", type=str, help="name of the g2p class"
)
@click.option(
    "--ap_detector",
    "-a",
    default="LoudnessSpectralcentroidAPDetector",  # "NoneAPDetector",
    type=str,
    help="name of the AP detector class",
)
@click.option(
    "--in_format",
    "-if",
    default="lab",
    required=False,
    type=str,
    help="File extension of input transcriptions. Default: lab",
)
@click.option(
    "--out_formats",
    "-of",
    default="textgrid,htk,trans",
    required=False,
    type=str,
    help="Types of output file, separated by comma. Supported types:"
         "textgrid(praat),"
         " htk(lab,nnsvs,sinsy),"
         " transcriptions.csv(diffsinger,trans,transcription,transcriptions)",
)
@click.option(
    "--save_confidence",
    "-sc",
    is_flag=True,
    default=False,
    show_default=True,
    help="save confidence.csv",
)
@click.option(
    "--dictionary",
    "-d",
    default="dictionary/opencpop-extension.txt",
    type=str,
    help="(only used when --g2p=='Dictionary') path to the dictionary",
)
def infer(onnx,
          folder,
          g2p,
          ap_detector,
          in_format,
          out_formats,
          save_confidence,
          **kwargs, ):
    config_file = pathlib.Path(onnx).with_name('config.yaml')
    assert os.path.exists(onnx), f"Onnx file does not exist: {onnx}"
    assert config_file.exists(), f"Config file does not exist: {config_file}"

    config = load_config_from_yaml(config_file)
    melspec_config = config['melspec_config']
    session = create_session(onnx)

    if not g2p.endswith("G2P"):
        g2p += "G2P"
    g2p_class = getattr(modules.g2p, g2p)
    grapheme_to_phoneme = g2p_class(**kwargs)
    out_formats = [i.strip().lower() for i in out_formats.split(",")]

    if not ap_detector.endswith("APDetector"):
        ap_detector += "APDetector"
    AP_detector_class = getattr(modules.AP_detector, ap_detector)
    get_AP = AP_detector_class(**kwargs)

    grapheme_to_phoneme.set_in_format(in_format)
    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(folder).rglob("*.wav"))
    predictions = []

    for i in tqdm(range(len(dataset)), desc="Processing", unit="sample"):
        wav_path, ph_seq, word_seq, ph_idx_to_word_idx = dataset[i]

        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform[0][None, :][0]
        if sr != melspec_config['sample_rate']:
            waveform = torchaudio.transforms.Resample(sr, melspec_config['sample_rate'])(waveform)

        wav_length = waveform.shape[0] / melspec_config["sample_rate"]
        ph_seq_id = np.array([config['vocab'][ph] for ph in ph_seq], dtype=np.int64)
        num_frames = int(
            (wav_length * melspec_config["scale_factor"] * melspec_config["sample_rate"] + 0.5) / melspec_config[
                "hop_length"]
        )
        results = run_inference(session, [waveform.numpy()], num_frames, [ph_seq_id])

        edge_diff = results['edge_diff']
        edge_prob = results['edge_prob']
        ph_prob_log = results['ph_prob_log']
        # ctc_logits = results['ctc_logits']
        T = results['T']

        ph_idx_seq, ph_time_int_pred, frame_confidence = decode(ph_seq_id, ph_prob_log, edge_prob, )
        total_confidence = np.exp(np.mean(np.log(frame_confidence + 1e-6)) / 3)

        # postprocess
        frame_length = melspec_config["hop_length"] / (
                melspec_config["sample_rate"] * melspec_config["scale_factor"]
        )
        ph_time_fractional = (edge_diff[ph_time_int_pred] / 2).clip(-0.5, 0.5)
        ph_time_pred = frame_length * (
            np.concatenate(
                [
                    ph_time_int_pred.astype("float32") + ph_time_fractional,
                    [T],
                ]
            )
        )
        ph_intervals = np.stack([ph_time_pred[:-1], ph_time_pred[1:]], axis=1)

        ph_seq_pred = []
        ph_intervals_pred = []
        word_seq_pred = []
        word_intervals_pred = []

        word_idx_last = -1
        for j, ph_idx in enumerate(ph_idx_seq):
            # ph_idx只能用于两种情况：ph_seq和ph_idx_to_word_idx
            if ph_seq[ph_idx] == "SP":
                continue
            ph_seq_pred.append(ph_seq[ph_idx])
            ph_intervals_pred.append(ph_intervals[j, :])

            word_idx = ph_idx_to_word_idx[ph_idx]
            if word_idx == word_idx_last:
                word_intervals_pred[-1][1] = ph_intervals[j, 1]
            else:
                word_seq_pred.append(word_seq[word_idx])
                word_intervals_pred.append([ph_intervals[j, 0], ph_intervals[j, 1]])
                word_idx_last = word_idx
        ph_seq_pred = np.array(ph_seq_pred)
        ph_intervals_pred = np.array(ph_intervals_pred).clip(min=0, max=None)
        word_seq_pred = np.array(word_seq_pred)
        word_intervals_pred = np.array(word_intervals_pred).clip(min=0, max=None)

        predictions.append((wav_path,
                            wav_length,
                            total_confidence,
                            ph_seq_pred,
                            ph_intervals_pred,
                            word_seq_pred,
                            word_intervals_pred))

    predictions = get_AP.process(predictions)
    predictions, log = post_processing(predictions)
    exporter = Exporter(predictions, log)

    if save_confidence:
        out_formats.append('confidence')

    exporter.export(out_formats)


if __name__ == '__main__':
    infer()
