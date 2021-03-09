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
"""An IDS (Ideographic Description Sequence)."""

from typing import Callable, Optional, Text, List, cast
from src.utils import region as region_lib

LookupCb = Callable[[Text], Optional[Text]]


def _fast_is_cjk(x) -> bool:
    """Given a character ordinal, returns whether or not it is CJK."""
    # Clauses, in order:
    # (1) CJK Unified Ideographs
    # (2) CJK Unified Ideographs Extension A
    # (3) CJK Unified Ideographs Extension B
    # (4) CJK Unified Ideographs Extension C
    # (5) CJK Unified Ideographs Extension D
    # (6) CJK Unified Ideographs Extension E
    # (7) CJK Unified Ideographs Extension F
    # (8) CJK Unified Ideographs Extension G
    return (0x4E00 <= x <= 0x9FFF) or (0x3400 <= x <= 0x4DBF) or (
        0x20000 <= x <= 0x2A5DF) or (0x2A700 <= x <= 0x2B73F) or (
            0x2B740 <= x <= 0x2B81F) or (0x2B820 <= x <= 0x2CEAF) or (
                0x2CEB0 <= x <= 0x2EBEF) or (0x30000 <= x <= 0x3134F)


class IdeographicSequence:
    """A character and IDS (Ideographic Description Sequence) decomposition.

   Dumb container for a UnicodeText-typed ideographic description sequence, i.e.
     想 => "⿱相心".
   Also implements a not-dumb recursively_expand method which can transform:
     想 => "⿱相心" => "⿱⿰木目心".

  Attributes:
    character: The character (u'你').
    decomposition: The IDS ("⿰亻尔").
  """

    def __init__(self,
                 character: Text,
                 decomposition: Optional[Text] = None,
                 lookup_cb: Optional[LookupCb] = None):
        """Inits a new Sequence.

    Provide either a known decomposition or a cb.

    Args:
      character: The character.
      decomposition: A known character decomposition.
      lookup_cb: A lookup callback for decomposing characters. If this callback
        fails, this method may raise a ValueError.

    Raises:
      ValueError: thrown if the |lookup_cb| fails to decompose |character|, or
      thrown if neither a |decomposition| nor a |lookup_cb| is provided.
    """
        self.character = character
        if decomposition is not None:
            self.decomposition = decomposition
        elif lookup_cb is not None:
            returned_decomposition = lookup_cb(character)
            if returned_decomposition is None:
                raise ValueError(f"No decomposition for {character}.")
            self.decomposition = cast(Text, returned_decomposition)
        else:
            raise ValueError("No provided decomposition or callback.")

    def _expand(self, decomposition: Text, lookup_cb: LookupCb):
        """Returns an expanded decomposition, using the lookup_cb.

    Args:
      decomposition: An already partially-decomposed character representation.
      lookup_cb: A lookup callback which may fail.

    Returns:
      An expanded decomposition, or the same decomposition if no expansion is
      possible.
    """
        output = []
        for c in decomposition:
            if not _fast_is_cjk(ord(c)):
                # Must be a verb, push this onto the stack as-is.
                output.append(c)
                continue

            expanded = lookup_cb(c)
            if expanded is None:
                # No decomposition, push this character onto the stack as-is
                output.append(c)
                continue
            elif expanded != c:
                # If the character had an expansion, let's recurse before
                # appending.
                output.extend(self._expand(expanded, lookup_cb))
            else:
                # Otherwise the character can be pushed as-is.
                output.append(c)
        return "".join(output)

    def recursively_expand(self, lookup_cb: LookupCb):
        """Given a lookup cb, expands this Sequence's decomposition in-place.

    Args:
      lookup_cb: A lookup callback which may fail.
    """
        self.decomposition = self._expand(self.decomposition, lookup_cb)


class IdeographicSequenceGroup:
    """A holder class for a character and its default/alternative decompositions.

  Attributes:
    default_sequence: The default Sequence for this IdeographicSequenceGroup.
  """

    def __init__(self, sequence: IdeographicSequence):
        """Instantiates a IdeographicSequenceGroup with a default for all regions.

    Args:
      sequence: A default IdeographicSequence.
    """
        self.default_sequence = sequence
        self._region_to_sequence_map = {}

    def insert(self, regions: List[region_lib.Region],
               alternative: IdeographicSequence):
        """Inserts an alternative sequence at each of a list of regions.

    Args:
      regions: A list of regions.
      alternative: An alternative IDS.

    Raises:
      ValueError: if the alternative sequence doesn't represent the same
      character as the default sequence.
    """
        if alternative.character != self.default_sequence.character:
            raise ValueError("Tried to insert a non-matching alternative IDS.")
        for r in regions:
            self._region_to_sequence_map[r] = alternative

    def character(self):
        """Returns the character this IdeographicSequenceGroup represents."""
        return self.default_sequence.character

    def get_sequence_at_region(
            self, input_region: region_lib.Region) -> IdeographicSequence:
        """Given a region, returns the corresponding sequence.

    Args:
      input_region: The input region.

    Returns:
      The corresponding sequence.
    """
        if input_region in self._region_to_sequence_map:
            return self._region_to_sequence_map[input_region]
        return self.default_sequence
