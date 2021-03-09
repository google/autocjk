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
"""Abstract class for storing decompositions and retrieving them."""

from absl import logging
from src.utils import ideographic_description_sequence
from src.utils import region
from src.utils import shape as shape_lib
from third_party import babelstone
from typing import Any, Dict, Iterable, List, Optional, Text, Tuple, Callable, TypeVar
import icu
import networkx as nx
import re
import sys

# The ids.txt database uses a string like  "^⿲冫虫𮫙$(K)" to list an IDS
# (ideographic description sequence) and its list of valid regions.
_IDS_AND_REGIONS_REGEX = re.compile(
    r"\^(?P<ids>\S+)\$\s*(\((?P<regions>[A-Z]+)\))?\s*")

VisualMetadataMap = Dict[Text, List[shape_lib.VisualMetadata]]

# Type variables for |_build_paths|.
K = TypeVar("K")
V = TypeVar("V")


def _build_paths(g: nx.DiGraph, node: Any, data_lookup: Callable[[Dict[K, V]],
                                                                 V],
                 accumulator: Callable[[V, V], V]):
    r"""Utility for accumulating the set of metadata along every path from a node.

  Example:

      -{f:'a'}-> 2 -{f:'b'}-> 3
     /
    1
     \
      -{f:'d'}-> 4 -{f:'e'}-> 5

  $ _build_paths(g, node=1, data_lookup=lambda d: d['f'], accumulator a, b: a+b)
  > { 2: 'a', 3: 'ba', 4: 'd', 5: 'de' }

  Args:
    g: The DiGraph in question.
    node: The source node
    data_lookup: The lambda which can accept a networkx-style metadata
      dictionary and extract a particular key (and, optionally, transform it).
    accumulator: The lambda which can accept two bits of metadata and join them
      in some arbitrary way.

  Returns:
    A map where the keys are nodes which appear below |node| in the graph, and
    where the values are accumulated results of the list of metadata items
    extracted by |data_lookup| from each edge along the path from |node| down to
    that node.
  """
    result = {}
    for succ in g.successors(node):
        if succ == node:
            continue
        result[succ] = data_lookup(g.get_edge_data(node, succ))
        for subnode, d in _build_paths(g, succ, data_lookup,
                                       accumulator).items():
            result[subnode] = accumulator(d, result[succ])
    return result


class NoDecompositionError(Exception):
    """Used when the Decomposer class has no decomposition for a character."""

    def __init__(self, character: Text):
        super().__init__(self)
        self.character = character

    def __str__(self):
        return f"[{self.character}] has no decomposition."


def _get_decomposition_and_regions(
        text_in: Text) -> Optional[Tuple[Text, Optional[Text]]]:
    """Given a string matching _IDS_AND_REGIONS_REGEX returns the components.

  Args:
    text_in: A string matching _IDS_AND_REGIONS_REGEX.

  Returns:
    A tuple of (a) the IDS string, and (b) a string-list of regions like
    "(ABCD)", or None on failure.

  """
    match = _IDS_AND_REGIONS_REGEX.match(text_in)
    if match is None:
        logging.debug("Didn't match compiled regex.")
        return None
    d = match.groupdict()
    return d["ids"], d["regions"]


def parse(
    text_in: Text
) -> Optional[ideographic_description_sequence.IdeographicSequenceGroup]:
    """Parses a line of text from //third_party/babelstone:babelstone_ids.

  Given a line of text from //third_party/babelstone:babelstone_ids, parses
  it and returns the resultant IdeographicSequenceGroup, if valid.
  Expects a tab-separated string where the second column is a single CJK
  character, and the third and all following columns match
  _IDS_AND_REGIONS_REGEX. For example:
  "U+4E05  丅      ^⿱一丨$(GJT)   ^下-㇔$(X)"

  Args:
    text_in: a line of text

  Returns:
    A resultant IdeographicSequenceGroup, or None if the input was invalid.
  """
    split_input = text_in.rstrip().split("\t")

    if len(split_input) < 3:
        logging.debug("Input must be at least three columns.")
        return None

    character = split_input[1]
    maybe_decomposition_and_regions = _get_decomposition_and_regions(
        split_input[2])
    if maybe_decomposition_and_regions is None:
        logging.debug("Invalid input: %s", text_in)
        return None
    default_decomposition, _ = maybe_decomposition_and_regions

    s_set = ideographic_description_sequence.IdeographicSequenceGroup(
        ideographic_description_sequence.IdeographicSequence(
            character, default_decomposition))

    for i in range(3, len(split_input)):
        maybe_decomposition_and_regions = _get_decomposition_and_regions(
            split_input[i])
        if maybe_decomposition_and_regions is None:
            return None
        alt_decomposition, alt_regions = maybe_decomposition_and_regions
        if not alt_regions:
            return None
        s_set.insert([region.Region[r] for r in alt_regions],
                     ideographic_description_sequence.IdeographicSequence(
                         character, alt_decomposition))

    return s_set


class Decomposer():
    """A class for holding sequences and retrieving them."""

    def __init__(self):

        # Graph where the nodes are unicode characters and the edges are "contains"
        # such that successors(尔) = [...你...]., and predecessors(你) = [亻,尔].
        # So, insert with self._graph.add_edge( "亻", "你" )
        #                 self._graph.add_edge( "尔", "你" )
        self._graph = nx.DiGraph()

        with open(babelstone.PATH_TO_IDS_TXT, encoding="UTF-8") as fp:
            for line in fp:
                # Ignore comments
                if line.startswith("#"):
                    continue
                # TODO(ambuc): ids.txt uses:
                # {1}, {2}, etc. to represent unencoded components.
                # ↔         as a mirror operator, i.e. to represent a component without
                #           a Unicode encoding, but whose mirror does have a Unicode
                #           encoding.
                # ↷        as a rotation operator, i.e. to represent a component
                #           without a Unicode encoding, but whose 180deg rotation does
                #           have a Unicode encoding.
                # 〾        as a variation indicator. We should try to handle these.
                # ?, ？     ids.txt uses these to represent an unencodable component.
                # We should probably try to handle these edge cases.
                elif re.search("[{}↔↷〾?？]", line):
                    continue

                maybe_parsed_set = parse(str(icu.UnicodeString(line)))
                if maybe_parsed_set is not None:
                    self.insert(maybe_parsed_set)

    def characters(self) -> Iterable[Text]:
        """Returns an iterable of characters which have decompositions."""
        return [
            node for node in self._graph.nodes()
            if list(self._graph.predecessors(node))
        ]

    def contains(self, character: Text) -> bool:
        """Returns True if a decomposer is aware of a character, False otherwise.

    Args:
      character: The query character.

    Returns:
      A boolean, true if the decomposer can handle the character.
    """
        if character not in self._graph.nodes():
            return False
        return bool(list(self._graph.predecessors(character)))

    def decompose(
        self, input_region: region.Region, character: Text
    ) -> ideographic_description_sequence.IdeographicSequence:
        """Attempts to decompose a character with respect to input region.

    Args:
      input_region: The input region. See region.py
      character: Some input character.

    Returns:
      An IdeographicSequence, if the character has a known decomposition
    Raises:
      NoDecompositionError: if the character has no known decomposition.
    """
        if character not in self._graph.nodes:
            raise NoDecompositionError(character)
        if "idg" not in self._graph.nodes[character]:
            raise NoDecompositionError(character)
        if not list(self._graph.predecessors(character)):
            raise NoDecompositionError(character)
        return self._graph.nodes[character]["idg"].get_sequence_at_region(
            input_region)

    def insert(
        self,
        sequence_set: ideographic_description_sequence.IdeographicSequenceGroup
    ) -> bool:
        """Inserts a character into the decomposer's internal map.

    TODO(ambuc): Support region-aware insertion.

    Args:
      sequence_set: the IdeographicSequenceGroup being inserted.

    Returns:
      True on success, False otherwise.
    """
        char = sequence_set.character()
        decomp = sequence_set.default_sequence.decomposition
        i = self._traverse_sequence(
            0, char, decomp,
            shape_lib.VisualMetadata(shape=shape_lib.UnitSquare, parent=char))
        if i < len(decomp):
            logging.debug("Something went wrong trying to parse decomp: %s",
                          ",".join(["U+%04x" % ord(o) for o in decomp]))
            return False

        self._graph.add_node(char, idg=sequence_set)
        return True

    def _get_with_component(
            self, component: Text) -> Iterable[Tuple[Text, shape_lib.Shape]]:
        return _build_paths(g=self._graph,
                            node=component,
                            data_lookup=lambda m: m["metadata"].shape,
                            accumulator=lambda a, b: a.portion(b)).items()

    def get_component(self, component: Text) -> List[Text]:
        return [c for c, _ in self._get_with_component(component)]

    def get_component_with_shape(self,
                                 component: Text,
                                 shape: shape_lib.Shape,
                                 w_t: float = 0.0,
                                 h_t: float = 0.0,
                                 x_t: float = 0.0,
                                 y_t: float = 0.0) -> List[Text]:
        """Returns characters containing |component| at |shape| in pure-space."""
        # pylint: disable=g-complex-comprehension
        return [
            c for c, summed_shape in self._get_with_component(component)
            if shape_lib.is_in_window(summed_shape.width, shape.width, w_t)
            and shape_lib.is_in_window(summed_shape.height, shape.height, h_t)
            and shape_lib.is_in_window(summed_shape.x_offset, shape.x_offset,
                                       x_t) and
            shape_lib.is_in_window(summed_shape.y_offset, shape.y_offset, y_t)
        ]
        # pylint: enable=g-complex-comprehension

    def get_component_with_aspectratio(self,
                                       component: Text,
                                       w: float = 0.0,
                                       h: float = 0.0,
                                       w_t: float = 0.0,
                                       h_t: float = 0.0) -> List[Text]:
        return [
            c for c, summed_shape in self._get_with_component(component)
            if shape_lib.is_in_window(summed_shape.width, w, w_t)
            and shape_lib.is_in_window(summed_shape.height, h, h_t)
        ]

    def get_shape_within(self, subcomponent: Text,
                         component: Text) -> shape_lib.Shape:
        """Returns the pure-space shape of |subcomponent| within |component|.

    Example usage:
      get_shape_within('尔', '你') => shape_lib.RightHalf

    Args:
      subcomponent: the subcomponent inside |component| we're seeking.
      component: a character which contains |subcomponent|.

    Returns:
      The pure-space shape of one within the other.
    """
        path = nx.shortest_path(self._graph, subcomponent, component)
        shape = shape_lib.UnitSquare
        while len(path) >= 2:
            shape = shape.portion(
                self._graph.get_edge_data(path[0], path[1])["metadata"].shape)
            path.pop(0)
        return shape

    def _traverse_sequence(self, i: int, character: Text, decomposition: Text,
                           metadata: shape_lib.VisualMetadata) -> int:
        """Given some partially-traversed decomposition AST, finishes traversing it.

    Args:
      i: The index in the decomposition our cursor is currently at.
      character: The character this tree corresponds to.
      decomposition: The decomposition AST, possibly partially traversed as
        indicated by |i|.
      metadata: A VisualMetadata struct representing the intermediate state of
        this node relative to the whole unit square.

    Returns:
      A new cursor position, or possibly the same cursor position if we are at
      the end.
    """
        if i >= len(decomposition):
            return i

        head = decomposition[i]
        i += 1

        # If there is no decomposition, we've reached a fundamental particle and
        # can't go any further.
        if not shape_lib.is_verb(head):
            self._graph.add_edge(head, character, metadata=metadata)
            return i

        for arg in shape_lib.get_subshapes_of_verb(head):
            i = self._traverse_sequence(
                i, character, decomposition,
                shape_lib.VisualMetadata(shape=metadata.shape.portion(arg),
                                         parent=character))

        return i
