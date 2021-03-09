# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ideographic_description_sequence."""

from typing import Optional, Text
from src.utils import region as region_lib
from unittest import mock
from src.utils import ideographic_description_sequence

from tensorflow.python.platform import googletest


class IdeographicDescriptionSequenceTest(googletest.TestCase):
    def test_construction(self):
        s = ideographic_description_sequence.IdeographicSequence("你", "⿰亻尔")
        self.assertEqual(s.character, "你")
        self.assertEqual(s.decomposition, "⿰亻尔")

    def test_construction_from_text(self):
        lookup_cb = mock.MagicMock()
        lookup_cb.return_value = "⿰言吾"

        s = ideographic_description_sequence.IdeographicSequence(
            "语", lookup_cb=lookup_cb)
        self.assertEqual(s.character, "语")
        self.assertEqual(s.decomposition, "⿰言吾")

        lookup_cb.assert_called_once_with("语")

    def test_invalid_construction_from_text(self):
        lookup_cb = mock.MagicMock()
        lookup_cb.return_value = None

        with self.assertRaises(ValueError) as _:
            ideographic_description_sequence.IdeographicSequence(
                "语", lookup_cb=lookup_cb)

        lookup_cb.assert_called_once_with(mock.ANY)

    def test_recursively_expand(self):
        s = ideographic_description_sequence.IdeographicSequence("语", "⿰言吾")
        self.assertEqual(s.character, "语")
        self.assertEqual(s.decomposition, "⿰言吾")

        def mini_parser(c: Text) -> Optional[Text]:
            if c == "言" or c == "五" or c == "口":
                return None
            elif c == "吾":
                return "⿱五口"

        lookup_cb = mock.MagicMock()
        lookup_cb.side_effect = mini_parser

        s.recursively_expand(lookup_cb)
        self.assertEqual(s.character, "语")
        self.assertEqual(s.decomposition, "⿰言⿱五口")

        # Idempotent.

        s.recursively_expand(lookup_cb)
        self.assertEqual(s.character, "语")
        self.assertEqual(s.decomposition, "⿰言⿱五口")


class IdeographicDescriptionSequenceSetTest(googletest.TestCase):
    def test_create_and_amend(self):
        char = "亟"
        decomp_default = "⿱⿻了叹一"
        decomp_alt = "⿱⿻丂叹一"

        s = ideographic_description_sequence.IdeographicSequenceGroup(
            sequence=ideographic_description_sequence.IdeographicSequence(
                char, decomp_default))
        self.assertEqual(s.character(), char)
        self.assertEqual(s.default_sequence.decomposition, decomp_default)
        self.assertEqual(
            s.get_sequence_at_region(region_lib.Region.H).decomposition,
            decomp_default)

        s.insert([region_lib.Region.H, region_lib.Region.J],
                 ideographic_description_sequence.IdeographicSequence(
                     char, decomp_alt))

        self.assertEqual(s.character(), char)
        self.assertEqual(s.default_sequence.decomposition, decomp_default)
        self.assertEqual(
            s.get_sequence_at_region(region_lib.Region.H).decomposition,
            decomp_alt)
        self.assertEqual(
            s.get_sequence_at_region(region_lib.Region.J).decomposition,
            decomp_alt)


if __name__ == '__main__':
    googletest.main()
