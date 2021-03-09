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
"""Tests for decomposer.py"""

from typing import cast

import more_itertools
import networkx as nx

from src.utils import decomposer
from src.utils import ideographic_description_sequence
from src.utils import region

from tensorflow.python.platform import googletest

# Only create this once. It is expensive.
DECOMPOSER_ = decomposer.Decomposer()


class GraphTest(googletest.TestCase):
    def test_graph(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, p="1 to 2")
        g.add_edge(1, 3, p="1 to 3")
        g.add_edge(2, 4, p="2 to 4")
        g.add_edge(4, 5, p="4 to 5")
        self.assertDictEqual(
            {
                2: "1 to 2",
                3: "1 to 3",
                4: "1 to 2, 2 to 4",
                5: "1 to 2, 2 to 4, 4 to 5",
            },
            decomposer._build_paths(g,
                                    node=1,
                                    data_lookup=lambda m: m["p"],
                                    accumulator=lambda a, b: b + ", " + a))


class DecomposerTest(googletest.TestCase):
    def test_construct_and_query(self):
        char = "你"
        sequence_default = "⿰亻尔"
        sequence_alt = "⿰人尔"

        s_set = ideographic_description_sequence.IdeographicSequenceGroup(
            ideographic_description_sequence.IdeographicSequence(
                char, sequence_default))
        s_set.insert([region.Region.B],
                     ideographic_description_sequence.IdeographicSequence(
                         char, sequence_alt))

        DECOMPOSER_.insert(s_set)

        self.assertTrue(DECOMPOSER_.contains(char))
        self.assertEqual(
            DECOMPOSER_.decompose(region.Region.G, char).decomposition,
            sequence_default)
        self.assertEqual(
            DECOMPOSER_.decompose(region.Region.B, char).decomposition,
            sequence_alt)

    def test_get_component_with_aspectratio(self):
        half_width_er = DECOMPOSER_.get_component_with_aspectratio("尔",
                                                                   w=0.5,
                                                                   h=1.0)
        self.assertContainsSubset(["你", "覙"], half_width_er)
        self.assertNotIn("玺", half_width_er)

        half_height_er = DECOMPOSER_.get_component_with_aspectratio("尔",
                                                                    w=1,
                                                                    h=0.5)
        self.assertNoCommonElements(["你", "覙"], half_height_er)
        self.assertIn("玺", half_height_er)

        self.assertEmpty(
            DECOMPOSER_.get_component_with_aspectratio("尔",
                                                       w=0.8,
                                                       w_t=0.0,
                                                       h=1.0,
                                                       h_t=0.0))
        self.assertContainsSubset(["你", "覙"],
                                  DECOMPOSER_.get_component_with_aspectratio(
                                      "尔", w=0.8, w_t=0.3, h=1.0, h_t=0.0))

    def test_get_component(self):
        # At all shapes and sizes.
        self.assertContainsSubset(["你", "您", "妳", "弥", "猕", "㟜", "㳽"],
                                  DECOMPOSER_.get_component("尔"))

    def test_parse_one_line(self):
        # NB: This line is tab-separated.
        line = ("U+4EE4\t令\t^⿱⿵𠆢丶龴$(G)\t^⿱⿵𠆢一龴$(HTV)\t" "^⿱⿵𠆢一𰆊$(JK)")

        maybe_ling_set = decomposer.parse(line)
        self.assertIsNotNone(maybe_ling_set)

        ling_set = cast(
            ideographic_description_sequence.IdeographicSequenceGroup,
            maybe_ling_set)
        self.assertEqual(ling_set.character(), "令")
        self.assertEqual(ling_set.default_sequence.decomposition, "⿱⿵𠆢丶龴")
        self.assertEqual(
            ling_set.get_sequence_at_region(region.Region.G).decomposition,
            "⿱⿵𠆢丶龴")

        for r in [region.Region.H, region.Region.T, region.Region.V]:
            self.assertEqual(
                ling_set.get_sequence_at_region(r).decomposition, "⿱⿵𠆢一龴")

        for r in [region.Region.J, region.Region.K]:
            self.assertEqual(
                ling_set.get_sequence_at_region(r).decomposition, "⿱⿵𠆢一𰆊")

    def test_thirdparty_database_contains(self):
        # Assert that the decomposer knows about some characters.
        self.assertTrue(DECOMPOSER_.contains("他"))
        self.assertTrue(DECOMPOSER_.contains("你"))
        self.assertTrue(DECOMPOSER_.contains("我"))
        # And that it didn't scoop up any garbage.
        self.assertFalse(DECOMPOSER_.contains("A"))
        self.assertFalse(DECOMPOSER_.contains(" "))
        self.assertFalse(DECOMPOSER_.contains("."))

    def test_thirdparty_database_decomposes(self):
        # Test some assorted decompositions.
        self.assertEqual(
            DECOMPOSER_.decompose(region.Region.G, "你").decomposition, "⿰亻尔")
        self.assertEqual(
            DECOMPOSER_.decompose(region.Region.G, "冒").decomposition, "⿱冃目")
        self.assertEqual(
            DECOMPOSER_.decompose(region.Region.G, "叹").decomposition, "⿰口又")
        self.assertEqual(
            DECOMPOSER_.decompose(region.Region.G, "亟").decomposition, "⿱⿻了叹一")

    # There are some characters we cannot decompose! Maybe ids.txt has a ? in
    # it.
    def test_thirdparty_database_cannot_decompose(self):
        with self.assertRaises(decomposer.NoDecompositionError) as _:
            # U+22016 ⿱ス工 contains a katakana (ス) for which ids.txt has no
            # representation.
            DECOMPOSER_.decompose(region.Region.G, "𢀖")

    def test_thirdparty_database_recursive_expansion(self):
        # Test recursive expansion
        sequence = DECOMPOSER_.decompose(region.Region.G, "厀")
        self.assertEqual(sequence.decomposition, "⿰桼卩")

        sequence.recursively_expand(
            lambda s: DECOMPOSER_.decompose(region.Region.G, s).decomposition)

        self.assertEqual(sequence.decomposition, "⿰⿱木⿱𠆢氺卩")


if __name__ == "__main__":
    googletest.main()
