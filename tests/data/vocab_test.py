import yaml

from src.torchsofa.data.vocab import Vocab


class TestVocab:
    def test_serialize(self):
        vocab = Vocab(
            ["a", "c", "c", "b", "a", "A", "B", "C", "D", "aa"],
            special_phones=["A", "B"],
            ignored_phones=["C", "D"],
            phone_aliases={"a": ["aa", "aaa"], "C": ["cc"]},
        )

        vocab.serialize("test.yaml")
        vocab_ = Vocab().deserialize("test.yaml")
        vocab_.serialize("test_.yaml")

        test = yaml.safe_load(open("test.yaml", "r"))
        test_ = yaml.safe_load(open("test_.yaml", "r"))
        assert test == test_, f"{test}!= {test_}"

    def test_get_id(self):
        vocab = Vocab(
            ["a", "c", "c", "b", "a"],
            special_phones=["A", "B"],
            ignored_phones=["C", "D"],
            phone_aliases={"a": ["aa", "aaa"], "C": ["cc"]},
        )

        assert vocab.get_id("c") == 2
        assert vocab.get_id("A") == 0
        assert vocab.get_id("D") == -1
        assert vocab.get_id("cc") == -1
        assert vocab.get_id("cashjkldfghwe") == -1

    def test_is_special(self):
        vocab = Vocab(
            ["a", "c", "c", "b", "a"],
            special_phones=["A", "B"],
            ignored_phones=["C", "D"],
            phone_aliases={"a": ["aa", "aaa"], "C": ["cc"]},
        )

        assert not vocab.is_special("a")
        assert vocab.is_special("B")
        assert not vocab.is_special("C")

    def test_get_phone(self):
        vocab = Vocab(
            ["a", "c", "c", "b", "a"],
            special_phones=["A", "B"],
            ignored_phones=["C", "D"],
            phone_aliases={"a": ["aa", "aaa"], "C": ["cc"]},
        )

        assert vocab.get_phone(0) == "a"
        assert vocab.get_phone(1, True) == "B"

    def test_get_ids(self):
        vocab = Vocab(
            ["a", "c", "c", "b", "a"],
            special_phones=["A", "B"],
            ignored_phones=["C", "D"],
            phone_aliases={"a": ["aa", "aaa"], "C": ["cc"]},
        )

        phones = ["c", "A", "D", "cc", "cashjkldfghwe"]
        ids = vocab.get_ids(phones)
        assert ids == [2, 0, -1, -1, -1]

    def test_is_specials(self):
        vocab = Vocab(
            ["a", "c", "c", "b", "a"],
            special_phones=["A", "B"],
            ignored_phones=["C", "D"],
            phone_aliases={"a": ["aa", "aaa"], "C": ["cc"]},
        )

        phones = ["c", "A", "D", "cc", "cashjkldfghwe"]
        specials = vocab.is_specials(phones)
        assert specials == [False, True, False, False, False]

    def test_get_phones(self):
        vocab = Vocab(
            ["a", "c", "c", "b", "a"],
            special_phones=["A", "B"],
            ignored_phones=["C", "D"],
            phone_aliases={"a": ["aa", "aaa"], "C": ["cc"]},
        )

        ids = [0, 1, 2, 0, 1]
        special = [False, False, False, True, True]
        phones = vocab.get_phones(ids, special)
        assert phones == ["a", "b", "c", "A", "B"]

    def test_summary(self):
        vocab = Vocab(
            ["a", "c", "c", "b", "a"],
            special_phones=["A", "B"],
            ignored_phones=["C", "D"],
            phone_aliases={"a": ["aa", "aaa"], "C": ["cc"]},
        )

        summary = vocab.summary()
        assert (
            summary
            == "normal phones: ['a', 'b', 'c']\nspecial phones: ['A', 'B']\nignored phones: ['C', 'D']\nphone aliases: {'a': ['aa', 'aaa'], 'C': ['cc']}"
        ), summary


if __name__ == "__main__":
    vocab = Vocab(
        ["a", "c", "c", "b", "a"],
        special_phones=["A", "B"],
        ignored_phones=["C", "D"],
        phone_aliases={"a": ["aa", "aaa"], "C": ["cc"]},
    )
