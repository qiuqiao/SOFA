import copy
import csv
import json
import os
import pathlib
import random
import librosa
import numpy as np
import torch
from scipy import interpolate
import modules.rmvpe
import yaml
from tqdm import tqdm

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
        self.config = config["preeprocessing"]
        self.config_global = config["global"]
        self.vocab = None
        self.meta_data_list = None

    def generate_vocab(self):
        print("generating vocab...")
        trans_path_list = [
            i
            for i in pathlib.Path(self.config["data_folder"]).rglob(
                "transcriptions.csv"
            )
            if i.name == "transcriptions.csv"
        ]

        phonemes = []
        for trans_path in tqdm(trans_path_list):
            with open(trans_path, "rt", newline="") as csvfile:
                reader = csv.reader(csvfile)
                data = [row for row in reader]
                columns = data[0]
                column_index = dict(zip(columns, range(len(columns))))
                rows = data[1:]
                for row in rows:
                    phonemes.extend(row[column_index["ph_seq"]].strip().split(" "))
        phonemes = set(phonemes)

        for p in self.config["ignore_phonemes"]:
            if p in phonemes:
                phonemes.remove(p)
        phonemes = sorted(phonemes)
        phonemes = ["<EMPTY>", *phonemes]

        vocab = dict(zip(phonemes, range(len(phonemes))))
        vocab.update(dict(zip(range(len(phonemes)), phonemes)))
        vocab.update({p: 0 for p in self.config["ignore_phonemes"]})
        vocab.update({"<vocab_size>": len(phonemes)})

        print(f"vocab_size is {len(phonemes)}")

        return vocab

    def split_train_valid_test(self, meta_data_list):
        pass

    def process(self):
        self.vocab = self.generate_vocab()
        with open(
            pathlib.Path(self.config["binary_data_folder"]) / "vocab.yaml", "w"
        ) as file:
            yaml.dump(self.vocab, file)

        # load meta data of each item
        self.meta_data_list = self.load_meta_data()
        # # split train, valid and test set
        # (
        #     self.meta_data_list_train,
        #     self.meta_data_list_valid,
        #     self.meta_data_list_test,
        # ) = self.split_train_valid_test(self.meta_data_list)
        # save test set
        # binarize and save valid set
        # binarize and save train set

    def load_meta_data(self):
        print("loading metadata...")
        path = pathlib.Path(self.config["data_folder"])
        trans_path_list = [
            i
            for i in path.rglob("transcriptions.csv")
            if i.name == "transcriptions.csv"
        ]
        wav_path_list = [i for i in path.rglob("*.wav")]
        meta_data_dict = {
            "wav_path": [],
            "ph_seq": [],
            "ph_dur": [],
        }

        # read data with transcriptions.csv
        for trans_path in trans_path_list:
            print(f"processing {trans_path}...")
            with open(trans_path, "rt", newline="") as csvfile:
                reader = csv.reader(csvfile)
                data = [row for row in reader]
                columns = data[0]
                column_index = dict(zip(columns, range(len(columns))))
                rows = np.array(data[1:])
                meta_data_dict["wav_path"].extend(
                    [
                        trans_path.parent / "wavs" / f"{i}.wav"
                        for i in rows[:, [column_index["name"]]]
                    ]
                )
                meta_data_dict["ph_seq"].extend(
                    rows[:, [column_index["ph_seq"]]].split(" ")
                )
        #         for row in tqdm(rows):
        #             meta_data = {}

        #             meta_data["wav_path"] = (
        #                 trans_path.parent / "wavs" / f"{row[column_index['name']]}.wav"
        #             )
        #             if meta_data["wav_path"] in wav_path_list:
        #                 wav_path_list.remove(meta_data["wav_path"])

        #             assert "ph_seq" in columns
        #             meta_data["ph_seq"] = np.array(
        #                 [
        #                     self.vocab[i]
        #                     for i in row[column_index["ph_seq"]].strip().split(" ")
        #                 ]
        #             )

        #             if "ph_dur" in columns:
        #                 meta_data["ph_dur"] = np.array(
        #                     [
        #                         float(i)
        #                         for i in row[column_index["ph_dur"]].strip().split(" ")
        #                     ]
        #                 )
        #                 assert len(meta_data["ph_dur"]) == len(meta_data["ph_seq"])
        #             else:
        #                 meta_data["ph_dur"] = None

        #             meta_data_list.append(meta_data)
        # print(wav_path_list)
        return meta_data_list
        # read data without transcriptions.csv

    #     transcription_data_list = list(raw_data_dir.rglob("transcriptions.csv"))

    #     for transcription_path in transcription_data_list:
    #         for utterance_label in csv.DictReader(
    #             open(transcription_path, "r", encoding="utf-8")
    #         ):
    #             item_name = utterance_label["name"]
    #             temp_dict = {
    #                 "wav_fn": str(
    #                     (transcription_path.parent / "wavs") / f"{item_name}.wav"
    #                 )
    #             }

    #             temp_dict["ph_seq"] = np.array(
    #                 [
    #                     self.config["vocab"][i]
    #                     for i in utterance_label["ph_seq"].strip().split(" ")
    #                 ]
    #             )
    #             if "ph_dur" in utterance_label:
    #                 temp_dict["ph_dur"] = np.array(
    #                     [float(i) for i in utterance_label["ph_dur"].strip().split(" ")]
    #                 )
    #             else:
    #                 temp_dict["ph_dur"] = np.zeros_like(
    #                     temp_dict["ph_seq"], dtype=np.float32
    #                 )
    #             if (
    #                 str(transcription_path.parent).find("strong_label") == -1
    #                 or "ph_dur" not in utterance_label
    #             ):
    #                 temp_dict["is_strong_label"] = False
    #             else:
    #                 temp_dict["is_strong_label"] = True

    #             meta_data_dict[f"{ds_id}:{item_name}"] = temp_dict

    #     self.items.update(meta_data_dict)

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
    with open("configs/config copy.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)
    ForcedAlignmentBinarizer(config=config).process()
