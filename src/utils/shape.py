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

import dataclasses
from typing import Text, List


def is_in_window(n, window_center, tolerance) -> bool:
    """Given a number and window, returns true if the number is in that range."""
    return window_center - tolerance <= n <= window_center + tolerance


#   x_offset
#   |---->|
#   |     |<--------->|      width
#   +-----+-----+-----+ ---
#   |     |     |     |  |
#   |     |     |     |  v   y_offset
#   +-----XXXXXXXXXXXXX ---
#   |     XXXXXXXXXXXXX  ^
#   |     XXXXXXXXXXXXX  |
#   +-----XXXXXXXXXXXXX  |   height
#   |     XXXXXXXXXXXXX  |
#   |     XXXXXXXXXXXXX  v
#   +-----XXXXXXXXXXXXX ---
@dataclasses.dataclass(frozen=True)
class Shape:
    """A holding dataclass for properties of a subslice of the unit square."""
    width: float
    height: float
    x_offset: float
    y_offset: float

    def within_shape(self,
                     shape: "Shape",
                     w_d: float = 0.1,
                     h_d: float = 0.1,
                     x_d: float = 0.1,
                     y_d: float = 0.1) -> bool:
        return is_in_window(self.height, shape.height, h_d) and is_in_window(
            self.width, shape.width, w_d) and is_in_window(
                self.x_offset, shape.x_offset, x_d) and is_in_window(
                    self.y_offset, shape.y_offset, y_d)

    def __repr__(self):
        return (f"Shape(width={self.width}, height={self.height}, "
                f"x_offset={self.x_offset}, y_offset={self.y_offset})")

    def __str__(self):
        return f"{self.width}x{self.height}@({self.x_offset},{self.y_offset})"

    @staticmethod
    def get(s: Text) -> "Shape":
        """Utility for turning a string into a predefined portion of the unit sq."""
        if s == "BottomHalf":
            return BottomHalf
        if s == "LeftHalf":
            return LeftHalf
        if s == "RightHalf":
            return RightHalf
        if s == "TopHalf":
            return TopHalf
        raise NotImplementedError()

    def portion(self, s: "Shape") -> "Shape":
        """Returns s scaled-down inside self.

    Returns a new shape which is the scaled-down version of |this| shape
    within not the unit square, but some |s| shape. Here is an example:
    #
    # self        // s           // self.portion(s) // s.portion(self)
    # +----XXXXXX // +----+----+ // +----+----+     // +----+----+
    # |    XXXXXX // |    |    | // |    |    |     // |    |    |
    # |    XXXXXX // |    |    | // |    |    |     // |    XXX  |
    # +----XXXXXX // XXXXXX----+ // +--XXX----+     // +----XXX--+
    # |    |    | // XXXXXX    | // |  XXX    |     // |    |    |
    # |    |    | // XXXXXX    | // |    |    |     // |    |    |
    # +----+----+ // XXXXXX----+ // +----+----+     // +----+----+

    Args:
      s: a shape

    Returns:
      s scaled-down inside self.
    """
        return Shape(width=self.width * s.width,
                     height=self.height * s.height,
                     x_offset=self.x_offset + (s.x_offset * self.width),
                     y_offset=self.y_offset + (s.y_offset * self.height))


UnitSquare = Shape(width=1, height=1, x_offset=0, y_offset=0)
LeftHalf = Shape(width=0.5, height=1, x_offset=0, y_offset=0)
RightHalf = Shape(width=0.5, height=1, x_offset=0.5, y_offset=0)
TopHalf = Shape(width=1, height=0.5, x_offset=0, y_offset=0)
BottomHalf = Shape(width=1, height=0.5, x_offset=0, y_offset=0.5)


@dataclasses.dataclass
class VisualMetadata:
    """A holding class for the shape of a subslice and the parent character."""
    shape: Shape
    parent: Text


_verbs = frozenset("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")


def is_verb(character: Text) -> bool:
    """Given a character, returns True if it is a verb.

  i.e. one of ⿰ ⿱ ⿲ ⿳ ⿴ ⿵ ⿶ ⿷ ⿸ ⿹ ⿺ ⿻

  Args:
    character: a CJK character, possibly a verb.

  Returns:
    A boolean indicating whether or not that character is a verb.
  """
    return character in _verbs


_SHAPES_BY_VERB = {
    "⿰": [LeftHalf, RightHalf],
    "⿲": [
        Shape(0.33, 1.0, 0.0, 0.0),
        Shape(0.33, 1.0, 0.33, 0.0),
        Shape(0.33, 1.0, 0.66, 0.0)
    ],
    "⿱": [TopHalf, BottomHalf],
    "⿳": [
        Shape(1.0, 0.33, 0.0, 0.0),
        Shape(1.0, 0.33, 0.0, 0.33),
        Shape(1.0, 0.33, 0.0, 0.66)
    ],
    # Both shapes are full-size. This is the tricky one and should probably
    # not be relied upon.
    "⿻": [UnitSquare, UnitSquare],
    # +----+----+----+----+
    # |    |    |    |    |
    # +----XXXXXXXXXXX----+
    # |    XXXXXXXXXXX    |
    # +----XXXXXXXXXXX----+
    # |    XXXXXXXXXXX    |
    # +----XXXXXXXXXXX----+
    # |    |    |    |    |
    # +----+----+----+----+
    "⿴": [
        Shape(1.0, 1.0, 0.0, 0.0),
        Shape(0.5, 0.5, 0.25, 0.25),
    ],
    # +----+----+----+----+
    # |    |    |    |    |
    # +----XXXXXXXXXXX----+
    # |    XXXXXXXXXXX    |
    # +----XXXXXXXXXXX----+
    # |    XXXXXXXXXXX    |
    # +----XXXXXXXXXXX----+
    # |    XXXXXXXXXXX    |
    # +----XXXXXXXXXXX----+
    "⿵": [
        Shape(1.0, 1.0, 0.0, 0.0),
        Shape(0.5, 0.75, 0.25, 0.25),
    ],
    # +----XXXXXXXXXXX----+
    # |    XXXXXXXXXXX    |
    # +----XXXXXXXXXXX----+
    # |    XXXXXXXXXXX    |
    # +----XXXXXXXXXXX----+
    # |    XXXXXXXXXXX    |
    # +----XXXXXXXXXXX----+
    # |    |    |    |    |
    # +----+----+----+----+
    "⿶": [
        Shape(1.0, 1.0, 0.0, 0.0),
        Shape(0.5, 0.75, 0.25, 0.0),
    ],
    # +----+----+----+----+
    # |    |    |    |    |
    # +----XXXXXXXXXXXXXXXX
    # |    XXXXXXXXXXXXXXXX
    # +----XXXXXXXXXXXXXXXX
    # |    XXXXXXXXXXXXXXXX
    # +----XXXXXXXXXXXXXXXX
    # |    |    |    |    |
    # +----+----+----+----+
    "⿷": [
        Shape(1.0, 1.0, 0.0, 0.0),
        Shape(0.75, 0.5, 0.25, 0.25),
    ],
    # +----+----+----+----+
    # |    |    |    |    |
    # +----XXXXXXXXXXXXXXXX
    # |    XXXXXXXXXXXXXXXX
    # +----XXXXXXXXXXXXXXXX
    # |    XXXXXXXXXXXXXXXX
    # +----XXXXXXXXXXXXXXXX
    # |    XXXXXXXXXXXXXXXX
    # +----XXXXXXXXXXXXXXXX
    "⿸": [
        Shape(1.0, 1.0, 0.0, 0.0),
        Shape(0.75, 0.75, 0.25, 0.25),
    ],
    # +----+----+----+----+
    # |    |    |    |    |
    # XXXXXXXXXXXXXXXX----+
    # XXXXXXXXXXXXXXXX    |
    # XXXXXXXXXXXXXXXX----+
    # XXXXXXXXXXXXXXXX    |
    # XXXXXXXXXXXXXXXX----+
    # XXXXXXXXXXXXXXXX    |
    # XXXXXXXXXXXXXXXX----+
    "⿹": [
        Shape(1.0, 1.0, 0.0, 0.0),
        Shape(0.75, 0.75, 0.0, 0.25),
    ],
    # +----XXXXXXXXXXXXXXXX
    # +    XXXXXXXXXXXXXXXX
    # +----XXXXXXXXXXXXXXXX
    # +    XXXXXXXXXXXXXXXX
    # +----XXXXXXXXXXXXXXXX
    # +    XXXXXXXXXXXXXXXX
    # +----XXXXXXXXXXXXXXXX
    # +    |    |    |    |
    # +----+----+----+----+
    "⿺": [
        Shape(1.0, 1.0, 0.0, 0.0),
        Shape(0.75, 0.75, 0.25, 0.0),
    ]
}


def get_subshapes_of_verb(verb: Text) -> List[Shape]:
    """Given a verb, returns a list of shapes describing the following args.

  Example: "⿰xy" means "x and y next to each other, each half-width". So
  get_subshapes_of_verb(⿰) returns a list of two shapes, each corresponding to
  the shape of the next argument.

  Args:
    verb: the input character.

  Returns:
    A list of shape structs describing the following arguments.

  Raises:
    ValueError: if the input verb is not a verb.
    KeyError: if the verb does not have a mapping in _SHAPES_BY_VERB.
  """
    if not is_verb(verb):
        raise ValueError(f"Input {verb.encode('UTF-8')} is not a verb.")
    return _SHAPES_BY_VERB[verb]


def num_subshapes_of_verb(verb: Text) -> int:
    """Given a verb, returns the number of subshapes.

    Args:
      verb: the input character.

    Returns:
      The number of subshapes of that verb

    Raises:
      ValueError: if the input verb is not a verb.
      KeyError: if the verb does not have a mapping in _SHAPES_BY_VERB.
    """
    return len(get_subshapes_of_verb(verb))
