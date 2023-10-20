import copy
import csv
import json
import pathlib
import random
import librosa
import numpy as np
import torch
from scipy import interpolate
import modules.rmvpe
import yaml
from tqdm import tqdm
import pandas as pd

FORCED_ALIGNER_ITEM_ATTRIBUTES = [
    "input_feature",  # contentvec units or mel spectroogram and putch, float32[T_s, input_feature_dim]
    "ph_seq",  # phoneme sequence, str[T_p,]
    "ph_edge",  # edge of phoneme, int32[T_s,], if the label does not exist, it is all -1
    "ph_frame",  # frame-wise phoneme class, int32[T_s,], if the label does not exist, it is all -1
]

mel_spec_transform = None
rmvpe = None


class ForcedAlignmentBinarizer:
    def __init__(self, config: dict):
        self.config_global = config["global"]

        self.data_folder_path = config["preprocessing"]["data_folder"]
        self.ignored_phonemes = config["preprocessing"]["ignored_phonemes"]
        self.binary_data_folder = config["preprocessing"]["binary_data_folder"]
        self.valid_set_size = config["preprocessing"]["valid_set_size"]
        self.data_folder = config["preprocessing"]["data_folder"]

    def get_vocab(self, data_folder_path, ignored_phonemes):
        print("generating vocab...")
        phonemes = []
        trans_path_list = pathlib.Path(data_folder_path).rglob("transcriptions.csv")

        for trans_path in trans_path_list:
            if trans_path.name == "transcriptions.csv":
                df = pd.read_csv(trans_path)
                ph = list(set(" ".join(df["ph_seq"]).split(" ")))
                phonemes.extend(ph)

        phonemes = set(phonemes)
        for p in ignored_phonemes:
            if p in phonemes:
                phonemes.remove(p)
        phonemes = sorted(phonemes)
        phonemes = ["<EMPTY>", *phonemes]

        vocab = dict(zip(phonemes, range(len(phonemes))))
        vocab.update(dict(zip(range(len(phonemes)), phonemes)))
        vocab.update({i: 0 for i in ignored_phonemes})
        vocab.update({"<vocab_size>": len(phonemes)})

        print(f"vocab_size is {len(phonemes)}")

        return vocab

    def process(self):
        vocab = self.get_vocab(self.data_folder_path, self.ignored_phonemes)
        with open(pathlib.Path(self.binary_data_folder) / "vocab.yaml", "w") as file:
            yaml.dump(vocab, file)

        # load meta data of each item
        meta_data_df = self.get_meta_data(self.data_folder)
        print(meta_data_df)

        # split train and valid set
        valid_set_size = int(self.valid_set_size)
        meta_data_valid = meta_data_df[meta_data_df["prefix"] != "no_label"].sample(
            valid_set_size
        )
        meta_data_train = meta_data_df.drop(meta_data_valid.index)

        # binarize valid set
        self.binarize(meta_data_valid)

        # binarize train set
        # self.binarize(meta_data_train)

    def binarize(self, meta_data: pd.DataFrame):
        meta_data["ph_seq"] = meta_data["ph_seq"].apply(lambda x: x.split(" "))

    def get_meta_data(self, data_folder):
        path = pathlib.Path(data_folder)
        trans_path_list = [
            i
            for i in path.rglob("transcriptions.csv")
            if i.name == "transcriptions.csv"
        ]

        print("loading metadata...")
        meta_data_df = pd.DataFrame()
        for trans_path in tqdm(trans_path_list):
            df = pd.read_csv(trans_path)
            df["path"] = df["name"].apply(
                lambda name: str(trans_path.parent / "wavs" / (str(name) + ".wav")),
            )
            df["prefix"] = df["path"].apply(
                lambda path: (
                    "strong_label"
                    if "strong_lasbel" in path
                    else "weak_label"
                    if "weak_label" in path
                    else "no_label"
                ),
            )
            meta_data_df = pd.concat([meta_data_df, df])

        no_label_df = pd.DataFrame(
            {"path": [i for i in (path / "no_label").rglob("*.wav")]}
        )
        meta_data_df = pd.concat([meta_data_df, no_label_df])
        meta_data_df["prefix"].fillna("no_label", inplace=True)

        meta_data_df.reset_index(drop=True, inplace=True)

        return meta_data_df

    # def _process_item(self, waveform, meta_data):
    #     wav_tensor = torch.from_numpy(waveform).to(self.device)
    #     units_encoder = self.config["units_encoder"]
    #     if units_encoder == "contentvec768l12":
    #         global contentvec_transform
    #         if contentvec_transform is None:
    #             contentvec_transform = modules.contentvec.ContentVec768L12(
    #                 self.config["units_encoder_ckpt"], device=self.device
    #             )
    #         units = contentvec_transform(wav_tensor).squeeze(0).cpu().numpy()
    #     elif units_encoder == "mel":
    #         global mel_spec_transform
    #         if mel_spec_transform is None:
    #             mel_spec_transform = modules.rmvpe.MelSpectrogram(
    #                 n_mel_channels=self.config["units_dim"],
    #                 sampling_rate=self.config["audio_sample_rate"],
    #                 win_length=self.config["win_size"],
    #                 hop_length=self.config["hop_size"],
    #                 mel_fmin=self.config["fmin"],
    #                 mel_fmax=self.config["fmax"],
    #             ).to(self.device)
    #         units = (
    #             mel_spec_transform(wav_tensor.unsqueeze(0))
    #             .transpose(1, 2)
    #             .squeeze(0)
    #             .cpu()
    #             .numpy()
    #         )
    #     else:
    #         raise NotImplementedError(f"Invalid units encoder: {units_encoder}")
    #     assert (
    #         len(units.shape) == 2 and units.shape[1] == self.config["units_dim"]
    #     ), f"Shape of units must be [T, units_dim], but is {units.shape}."
    #     length = units.shape[0]
    #     seconds = length * self.config["hop_size"] / self.config["audio_sample_rate"]
    #     processed_input = {"seconds": seconds, "length": length, "units": units}

    #     f0_algo = self.config["pe"]
    #     if f0_algo == "parselmouth":
    #         f0, _ = get_pitch_parselmouth(
    #             waveform,
    #             sample_rate=self.config["audio_sample_rate"],
    #             hop_size=self.config["hop_size"],
    #             length=length,
    #             interp_uv=True,
    #         )
    #     elif f0_algo == "rmvpe":
    #         global rmvpe
    #         if rmvpe is None:
    #             rmvpe = modules.rmvpe.RMVPE(self.config["pe_ckpt"], device=self.device)
    #         f0, _ = rmvpe.get_pitch(
    #             waveform,
    #             sample_rate=self.config["audio_sample_rate"],
    #             hop_size=rmvpe.mel_extractor.hop_length,
    #             length=(waveform.shape[0] + rmvpe.mel_extractor.hop_length - 1)
    #             // rmvpe.mel_extractor.hop_length,
    #             interp_uv=True,
    #         )
    #         f0 = resample_align_curve(
    #             f0,
    #             original_timestep=rmvpe.mel_extractor.hop_length
    #             / self.config["audio_sample_rate"],
    #             target_timestep=self.config["hop_size"]
    #             / self.config["audio_sample_rate"],
    #             align_length=length,
    #         )
    #     else:
    #         raise NotImplementedError(f"Invalid pitch extractor: {f0_algo}")
    #     pitch = librosa.hz_to_midi(f0)
    #     processed_input["pitch"] = pitch

    #     ph_seq = np.array(meta_data["ph_seq"])
    #     processed_input["ph_seq"] = ph_seq

    #     ph_dur = np.array(meta_data["ph_dur"])
    #     ph_edge = np.zeros(length, dtype="int32")
    #     if meta_data["is_strong_label"] == True:
    #         ph_edge[(ph_dur.cumsum() / self.timestep).round().astype("int32")[:-1]] = 1
    #         ph_edge = ph_edge
    #     else:
    #         ph_edge = ph_edge - 1
    #     processed_input["ph_edge"] = ph_edge

    #     if meta_data["is_strong_label"] == True:
    #         ph_frame = ph_seq[ph_edge.cumsum()]
    #         # torch.gather(
    #         #     torch.tensor(ph_seq), 0, torch.tensor(ph_edge).cumsum()
    #         # )
    #         processed_input["ph_frame"] = ph_frame
    #     else:
    #         processed_input["ph_frame"] = np.zeros_like(ph_edge) - 1

    #     return processed_input

    # @torch.no_grad()
    # def process_item(self, item_name, meta_data, allow_aug=False):
    #     waveform, _ = librosa.load(
    #         meta_data["wav_fn"], sr=self.config["audio_sample_rate"], mono=True
    #     )

    #     processed_input = self._process_item(waveform, meta_data)
    #     items = [processed_input]
    #     if not allow_aug:
    #         return items

    #     wav_tensor = torch.from_numpy(waveform).to(self.device)
    #     for _ in range(self.config["key_shift_factor"]):
    #         assert (
    #             mel_spec_transform is not None
    #         ), "Units encoder must be mel if augmentation is applied!"
    #         key_shift = (
    #             random.random() * (self.key_shift_max - self.key_shift_min)
    #             + self.key_shift_min
    #         )
    #         processed_input_aug = copy.deepcopy(processed_input)
    #         assert isinstance(mel_spec_transform, modules.rmvpe.MelSpectrogram)
    #         processed_input_aug["units"] = (
    #             mel_spec_transform(wav_tensor.unsqueeze(0), keyshift=key_shift)
    #             .transpose(1, 2)
    #             .squeeze(0)
    #             .cpu()
    #             .numpy()
    #         )
    #         processed_input_aug["pitch"] += key_shift
    #         items.append(processed_input_aug)

    #     return items


if __name__ == "__main__":
    with open("configs/config_new.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)
    ForcedAlignmentBinarizer(config=config).process()
