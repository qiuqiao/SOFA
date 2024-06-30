import numpy as np
import pandas as pd
import textgrid


class Exporter:
    def __init__(self, predictions, log):
        self.predictions = predictions
        self.log = log

    def save_textgrids(self):
        print("Saving TextGrids...")

        for (
                wav_path,
                wav_length,
                confidence,
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
        ) in self.predictions:
            tg = textgrid.TextGrid()
            word_tier = textgrid.IntervalTier(name="words")
            ph_tier = textgrid.IntervalTier(name="phones")

            for word, (start, end) in zip(word_seq, word_intervals):
                word_tier.add(start, end, word)

            for ph, (start, end) in zip(ph_seq, ph_intervals):
                ph_tier.add(minTime=float(start), maxTime=end, mark=ph)

            tg.append(word_tier)
            tg.append(ph_tier)

            label_path = (
                    wav_path.parent / "TextGrid" / wav_path.with_suffix(".TextGrid").name
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            tg.write(label_path)

    def save_htk(self):
        print("Saving htk labels...")

        for (
                wav_path,
                wav_length,
                confidence,
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
        ) in self.predictions:
            label = ""
            for ph, (start, end) in zip(ph_seq, ph_intervals):
                start_time = int(float(start) * 10000000)
                end_time = int(float(end) * 10000000)
                label += f"{start_time} {end_time} {ph}\n"
            label_path = (
                    wav_path.parent / "htk" / "phones" / wav_path.with_suffix(".lab").name
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(label)
                f.close()

            label = ""
            for word, (start, end) in zip(word_seq, word_intervals):
                start_time = int(float(start) * 10000000)
                end_time = int(float(end) * 10000000)
                label += f"{start_time} {end_time} {word}\n"
            label_path = (
                    wav_path.parent / "htk" / "words" / wav_path.with_suffix(".lab").name
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(label)
                f.close()

    def save_transcriptions(self):
        print("Saving transcriptions.csv...")

        folder_to_data = {}

        for (
                wav_path,
                wav_length,
                confidence,
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
        ) in self.predictions:
            folder = wav_path.parent
            if folder in folder_to_data:
                curr_data = folder_to_data[folder]
            else:
                curr_data = {
                    "name": [],
                    "word_seq": [],
                    "word_dur": [],
                    "ph_seq": [],
                    "ph_dur": [],
                }

            name = wav_path.with_suffix("").name
            word_seq = " ".join(word_seq)
            ph_seq = " ".join(ph_seq)
            word_dur = []
            ph_dur = []

            last_word_end = 0
            for start, end in word_intervals:
                dur = np.round(end - last_word_end, 5)
                word_dur.append(dur)
                last_word_end += dur

            last_ph_end = 0
            for start, end in ph_intervals:
                dur = np.round(end - last_ph_end, 5)
                ph_dur.append(dur)
                last_ph_end += dur

            word_dur = " ".join([str(i) for i in word_dur])
            ph_dur = " ".join([str(i) for i in ph_dur])

            curr_data["name"].append(name)
            curr_data["word_seq"].append(word_seq)
            curr_data["word_dur"].append(word_dur)
            curr_data["ph_seq"].append(ph_seq)
            curr_data["ph_dur"].append(ph_dur)

            folder_to_data[folder] = curr_data

        for folder, data in folder_to_data.items():
            df = pd.DataFrame(data)
            path = folder / "transcriptions"
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            df.to_csv(path / "transcriptions.csv", index=False)

    def save_confidence_fn(self):
        print("saving confidence...")

        folder_to_data = {}

        for (
                wav_path,
                wav_length,
                confidence,
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
        ) in self.predictions:
            folder = wav_path.parent
            if folder in folder_to_data:
                curr_data = folder_to_data[folder]
            else:
                curr_data = {
                    "name": [],
                    "confidence": [],
                }

            name = wav_path.with_suffix("").name
            curr_data["name"].append(name)
            curr_data["confidence"].append(confidence)

            folder_to_data[folder] = curr_data

        for folder, data in folder_to_data.items():
            df = pd.DataFrame(data)
            path = folder / "confidence"
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            df.to_csv(path / "confidence.csv", index=False)

    def export(self, out_formats):
        if "textgrid" in out_formats or "praat" in out_formats:
            self.save_textgrids()
        if (
                "htk" in out_formats
                or "lab" in out_formats
                or "nnsvs" in out_formats
                or "sinsy" in out_formats
        ):
            self.save_htk()
        if (
                "trans" in out_formats
                or "transcription" in out_formats
                or "transcriptions" in out_formats
                or "transcriptions.csv" in out_formats
                or "diffsinger" in out_formats
        ):
            self.save_transcriptions()

        if "confidence" in out_formats:
            self.save_confidence_fn()

        if self.log:
            print("error:")
            for line in self.log:
                print(line)
