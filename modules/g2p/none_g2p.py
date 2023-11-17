import numpy as np

from modules.g2p.base_g2p import BaseG2P


class NoneG2P(BaseG2P):
    def __init__(self, **kwargs):
        pass

    def _g2p(self, input_text):
        input_seq = input_text.strip().split(" ")

        ph_seq = ["SP"]
        for i, ph in enumerate(input_seq):
            if ph == "SP" and ph_seq[-1] == "SP":
                continue
            ph_seq.append(ph)
        if ph_seq[-1] != "SP":
            ph_seq.append("SP")

        word_seq = ph_seq
        ph_idx_to_word_idx = np.arange(len(ph_seq))

        return ph_seq, word_seq, ph_idx_to_word_idx


if __name__ == "__main__":
    pass
    grapheme_to_phoneme = NoneG2P()
    text = "wo shi SP yi ge xue sheng"
    ph_seq, word_seq, ph_idx_to_word_idx = grapheme_to_phoneme(text)
    print(ph_seq)
    print(word_seq)
    print(ph_idx_to_word_idx)
