import importlib

import librosa
import torch


def check_and_import(package_name):
    try:
        module = importlib.import_module(package_name)
        globals()[package_name] = importlib.import_module(package_name)
        print(f"'{package_name}' installed and imported.")
        return True, module
    except ImportError:
        print(f"'{package_name}' not installed.")
        return False, None


installed_torchaudio, torchaudio = check_and_import("torchaudio")
resample_transform_dict = {}


def load_wav(path, device, sample_rate=None):
    global installed_torchaudio
    if installed_torchaudio:
        waveform, sr = torchaudio.load(str(path))
        if sample_rate != sr and sample_rate is not None:
            global resample_transform_dict
            if sr not in resample_transform_dict:
                resample_transform_dict[sr] = torchaudio.transforms.Resample(
                    sr, sample_rate
                )

            waveform = resample_transform_dict[sr](waveform)

        waveform = waveform[0].to(device)

    else:
        waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
        waveform = torch.from_numpy(waveform).to(device)

    return waveform
