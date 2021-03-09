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
"""A class for handling font file access."""

from typing import Text, List, Iterable

from fontTools import ttLib
from fontTools.ttLib import ttFont

from src.utils import region as region_lib

_RANGES = [
    # (1) CJK Unified Ideographs
    (0x4E00, 0x9FFF),
    # (2) CJK Unified Ideographs Extension A
    (0x3400, 0x4DBF),
    # (3) CJK Unified Ideographs Extension B
    (0x20000, 0x2A5DF),
    # (4) CJK Unified Ideographs Extension C
    (0x2A700, 0x2B73F),
    # (5) CJK Unified Ideographs Extension D
    (0x2B740, 0x2B81F),
    # (6) CJK Unified Ideographs Extension E
    (0x2B820, 0x2CEAF),
    # (7) CJK Unified Ideographs Extension F
    (0x2CEB0, 0x2EBEF),
    # (8) CJK Unified Ideographs Extension G
    (0x30000, 0x3134F),
]


def _fast_is_cjk(x) -> bool:
    """Given a character ordinal, returns whether or not it is CJK."""
    return any(l <= x <= r for l, r in _RANGES)


def _cjk_characters() -> Iterable[Text]:
    """An iterator of all CJK codepoints."""
    for l, r in _RANGES:
        for i in range(l, r):
            yield chr(i)


class FontHelper:
    def __init__(self, font: ttLib.TTFont, input_region: region_lib.Region):
        self.font = font
        self.region = input_region
        self.glyph_set = self.font.getGlyphSet(preferCFF=True)
        self._cmap = self.font.getBestCmap()

    def has_ttglyph(self, character: Text) -> bool:
        """Returns True if self.font has a rendering for this character."""
        if len(character) > 1:
            # If this is an IDS, let's assume we don't have it.
            return False

        return ord(character) in self._cmap and self._cmap[ord(
            character)] in self.glyph_set

    def unknown_characters(self) -> Iterable[Text]:
        """Returns an iterator of CJK ideographic characters NOT in this font."""
        for c in _cjk_characters():
            if ord(c) not in self._cmap.keys():
                yield c

    def known_characters(self) -> List[Text]:
        """Returns the list of CJK ideographic characters in this font."""

        return list(map(chr, filter(_fast_is_cjk, self._cmap.keys())))
