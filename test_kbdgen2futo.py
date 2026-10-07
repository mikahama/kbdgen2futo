#!/usr/bin/env python3
"""Tests for the kbdgen Android-layout converter."""
import unittest
from pathlib import Path

import kbdgen2futo as converter


class LayoutConverterTests(unittest.TestCase):
    def test_uses_android_layers_and_preserves_longpress_per_case(self):
        source = {
            "displayNames": {"en": "Test"},
            "android": {"primary": {"layers": {"default": "a b", "shift": "A B"}}},
            "iOS": {"primary": {"layers": {"default": "x y"}}},
            "longpress": {"a": "á à", "A": "Á À", "B": "Ḃ"},
        }
        result = converter.convert_layout(source, Path("test.yaml"), None, "xx")
        self.assertEqual(result["rows"][0]["letters"], [
            {"type": "case", "normal": ["a", "á", "à"], "shifted": ["A", "Á", "À"]},
            {"type": "case", "normal": "b", "shifted": ["B", "Ḃ"]},
        ])
        self.assertEqual(result["attributes"], {"moreKeyMode": "OnlyExplicit"})

    def test_special_keys_do_not_gain_morekeys(self):
        source = {"android": {"primary": {"layers": {"default": r"\s{shift} a"}}},
                  "longpress": {r"\s{shift}": "x"}}
        result = converter.convert_layout(source, Path("test.yaml"), None, "xx")
        self.assertEqual(result["rows"][0]["letters"][0], "$shift")


if __name__ == "__main__":
    unittest.main()
