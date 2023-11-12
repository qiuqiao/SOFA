from modules.g2p.base_g2p import BaseG2P


class DictionaryG2P(BaseG2P):
    def __init__(self, *args):
        dict_path = args[0]
        with open(dict_path, 'r') as f:
            dictionary = f.read().strip().split('\n')
        self.dictionary = {item.split('\t')[0].strip(): item.split('\t')[1].strip().split(' ') for item in dictionary}

    def _g2p(self, input_text):
        word_seq = input_text.strip().split(' ')
        ph_seq = ['SP']
        ph_idx_to_word_idx = [-1]
        for word_idx, word in enumerate(word_seq):
            for ph in self.dictionary[word]:
                ph_seq.append(ph)
                ph_idx_to_word_idx.append(word_idx)
            ph_seq.append('SP')
            ph_idx_to_word_idx.append(-1)
        return ph_seq, word_seq, ph_idx_to_word_idx


if __name__ == "__main__":
    pass
    # grapheme_to_phoneme = DictionaryG2P('/home/qq/Project/SOFA/dictionary/opencpop-extension.txt')
    # text = 'wo shi yi ge xue sheng '
    # ph_seq, word_seq, ph_idx_to_word_idx = grapheme_to_phoneme(text)
    # print(ph_seq)
    # print(word_seq)
    # print(ph_idx_to_word_idx)
