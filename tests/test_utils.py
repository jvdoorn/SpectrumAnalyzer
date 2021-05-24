import unittest
from typing import Any

from specc.utils import is_list_of


class TestIsListOf(unittest.TestCase):
    def test_non_list(self):
        _obj = "a"
        self.assertFalse(is_list_of(_obj, Any))

    def test_empty_list(self):
        _obj = []
        self.assertTrue(is_list_of(_obj, Any))

    def test_list_of_correct_type(self):
        _obj = ["a", "b", "c"]
        self.assertTrue(is_list_of(_obj, str))

    def test_list_of_incorrect_type(self):
        _obj = ["a", "b", "c"]
        self.assertFalse(is_list_of(_obj, int))

    def test_list_of_mixed_type(self):
        _obj = ["a", "b", 1]
        self.assertFalse(is_list_of(_obj, int))


if __name__ == '__main__':
    unittest.main()
