import json
import warnings
from typing import Dict, List, Union

import numpy as np


class Vocab:
    # __slots__ = [
    #     "size",
    #     "size_special",
    #     "ignored_phones",
    #     "special_phones",
    #     "phone_to_id",
    #     "_id_to_phone",
    # ]

    def __init__(
        self,
        all_phones: Union[List[str], set] = None,
        special_phones: Union[List[str], set] = None,
        ignored_phones: Union[List[str], set] = None,
        phone_aliases: Dict[str, List[str]] = None,
    ):
        super().__init__()
        self.normal_phone_to_id = {}
        self.special_phone_to_id = {}
        self.ignored_phones = []
        self.phone_aliases = {}
        aliases = []

        if special_phones is not None:
            special_phones = list(set(special_phones))
            special_phones.sort()
            self.special_phone_to_id = {
                phone: id for id, phone in enumerate(special_phones)
            }

        if ignored_phones is not None:
            ignored_phones = list(set(ignored_phones))
            ignored_phones.sort()
            self.ignored_phones = ignored_phones

        if phone_aliases is not None:
            self.phone_aliases = phone_aliases
            aliases = [
                alias for _, aliases in self.phone_aliases.items() for alias in aliases
            ]

        if all_phones is not None:
            all_phones = set(all_phones)
            normal_phones = (
                all_phones
                - set(self.special_phone_to_id.keys())
                - set(self.ignored_phones)
                - set(aliases)
            )
            normal_phones = list(set(normal_phones))
            normal_phones.sort()
            self.normal_phone_to_id = {
                phone: id for id, phone in enumerate(normal_phones)
            }

        self._init()

    def _init(self):
        self._size = len(self.normal_phone_to_id)

        self._size_special = len(self.special_phone_to_id)

        self._phone_to_id = {}
        self._phone_to_id.update(self.normal_phone_to_id)
        self._phone_to_id.update(self.special_phone_to_id)

        self._id_to_normal_phone = {
            id: phone for phone, id in self.normal_phone_to_id.items()
        }
        self._id_to_special_phone = {
            id: phone for phone, id in self.special_phone_to_id.items()
        }

        self._alias_to_phone = {
            alias: phone
            for phone, aliases in self.phone_aliases.items()
            for alias in aliases
        }

    def serialize(self, file_path: str) -> None:
        data = {
            "normal_phone_to_id": self.normal_phone_to_id,
            "special_phone_to_id": self.special_phone_to_id,
            "ignored_phones": self.ignored_phones,
            "phone_aliases": self.phone_aliases,
        }
        # yaml.safe_dump(data, open(file_path, "w", encoding="utf-8"), sort_keys=False)
        json.dump(data, open(file_path, "w", encoding="utf-8"), sort_keys=False)

    def deserialize(self, file_path: str) -> None:
        # data = yaml.safe_load(open(file_path, "r", encoding="utf-8"))
        data = json.load(open(file_path, "r", encoding="utf-8"))
        self.normal_phone_to_id = data["normal_phone_to_id"]
        self.special_phone_to_id = data["special_phone_to_id"]
        self.ignored_phones = data["ignored_phones"]
        self.phone_aliases = data["phone_aliases"]
        self._init()
        return self

    def get_id(self, phone: str) -> int:
        if phone in self.ignored_phones:
            return -1
        if phone in self._alias_to_phone:
            return self.get_id(self._alias_to_phone[phone])
        if phone in self._phone_to_id:
            return self._phone_to_id[phone]
        warnings.warn(f"Phone {phone} not found in vocab. Ignore.")
        return -1

    def is_special(self, phone: str) -> bool:
        """
        Check if a token is special.

        Args:
            phone (str): A token.

        Returns:
            bool: True if the token is special, False otherwise.
        """
        return phone in self.special_phone_to_id

    def get_phone(self, id: int, special: bool = False) -> str:
        if special:
            return self._id_to_special_phone[id]
        return self._id_to_normal_phone[id]

    def get_ids(self, phones: List[str]) -> List[int]:
        """
        Convert a list of tokens to a list of IDs.

        Args:
            phones (List[str]): A list of tokens.

        Returns:
            List[int]: A list of IDs corresponding to the given tokens.
        """
        return [self.get_id(phone) for phone in phones]

    def is_specials(self, phones: List[str]) -> List[bool]:
        """
        Check if a list of tokens are special.

        Args:
            phones (List[str]): A list of tokens.

        Returns:
            List[bool]: A list of booleans indicating whether each token is special.
        """
        return [self.is_special(phone) for phone in phones]

    def get_phones(
        self, ids: Union[List[int], np.ndarray], specials: List[bool]
    ) -> List[str]:
        """
        Convert a list or numpy array of IDs to a list of tokens.

        Args:
            ids (List[int] | np.ndarray): A list or numpy array of IDs.
            specials(List[bool]): A list of booleans indicating whether each ID is special.

        Returns:
            List[str]: A list of tokens corresponding to the given IDs.
        """
        assert len(ids) == len(
            specials
        ), f"IDs and specials must have the same length, but got {len(ids)} and {len(specials)}."
        return [self.get_phone(id, special) for id, special in zip(ids, specials)]

    def summary(self) -> str:
        """
        Return a summary of the vocabulary.

        Returns:
            str: A summary of the vocabulary.
        """
        return "normal phones: {}\nspecial phones: {}\nignored phones: {}\nphone aliases: {}".format(
            list(self.normal_phone_to_id.keys()),
            list(self.special_phone_to_id.keys()),
            self.ignored_phones,
            self.phone_aliases,
        )
