from modules.g2p.base_g2p import BaseG2P


class PhonemeG2P(BaseG2P):
    def __init__(self, **kwargs):
        pass

    def _g2p(self, input_text):
        word_seq = input_text.strip().split(" ")
        word_seq = [ph for ph in word_seq if ph != "SP"]
        ph_seq = ["SP"]
        ph_idx_to_word_idx = [-1]
        for word_idx, word in enumerate(word_seq):
            ph_seq.append(word)
            ph_idx_to_word_idx.append(word_idx)
            ph_seq.append("SP")
            ph_idx_to_word_idx.append(-1)
        return ph_seq, word_seq, ph_idx_to_word_idx


if __name__ == "__main__":
    pass
    grapheme_to_phoneme = PhonemeG2P()
    text = "wo shi yi ge xue sheng SP SP SP"
    ph_seq, word_seq, ph_idx_to_word_idx = grapheme_to_phoneme(text)
    print(ph_seq)
    print(word_seq)
    print(ph_idx_to_word_idx)
