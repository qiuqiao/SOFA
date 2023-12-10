import warnings

from modules.g2p.base_g2p import BaseG2P


class DictionaryG2P(BaseG2P):
    def __init__(self, **kwargs):
        dict_path = kwargs["dictionary"]
        with open(dict_path, "r") as f:
            dictionary = f.read().strip().split("\n")
        self.dictionary = {
            item.split("\t")[0].strip(): item.split("\t")[1].strip().split(" ")
            for item in dictionary
        }

    def _g2p(self, input_text):
        word_seq_raw = input_text.strip().split(" ")
        word_seq = []
        word_seq_idx = 0
        ph_seq = ["SP"]
        ph_idx_to_word_idx = [-1]
        for word in word_seq_raw:
            if word not in self.dictionary:
                warnings.warn(f"Word {word} is not in the dictionary. Ignored.")
                continue
            word_seq.append(word)
            phones = self.dictionary[word]
            for i, ph in enumerate(phones):
                if (i == 0 or i == len(phones) - 1) and ph == "SP":
                    warnings.warn(
                        f"The first or last phoneme of word {word} is SP, which is not allowed. "
                        "Please check your dictionary."
                    )
                    continue
                ph_seq.append(ph)
                ph_idx_to_word_idx.append(word_seq_idx)
            if ph_seq[-1] != "SP":
                ph_seq.append("SP")
                ph_idx_to_word_idx.append(-1)
            word_seq_idx += 1

        return ph_seq, word_seq, ph_idx_to_word_idx


if __name__ == "__main__":
    pass
    grapheme_to_phoneme = DictionaryG2P(
        **{"dictionary": "/home/qq/Project/SOFA/dictionary/opencpop-extension.txt"}
    )
    text = "wo SP shi yi ge xue sheng a"
    ph_seq, word_seq, ph_idx_to_word_idx = grapheme_to_phoneme(text)
    print(ph_seq)
    print(word_seq)
    print(ph_idx_to_word_idx)
