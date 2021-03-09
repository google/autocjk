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
r"""Generate training data.

For usage, see README.md.
"""

from absl import app
from absl import flags
from absl import logging
from fontTools import ttLib
from multiprocessing.pool import ThreadPool
from typing import Sequence
import imageutils
import os
import re

from src.utils import font_helper as font_helper_lib
from src.utils import decomposer as decomposer_lib
from src.utils import ideographic_description_sequence
from src.utils import region as region_lib

_FONT_PATHS = flags.DEFINE_list("fonts", [], "List of font files.")
_DEST = flags.DEFINE_string(
    "dst", None, "Path to a directory into which to write. Required.")
_LIMIT = flags.DEFINE_integer(
    "limit", None,
    "If set, maximum number of training pairs to produce. Useful when testing."
)
_NUM_WORKERS = flags.DEFINE_integer(
    "num_workers", 40,
    "Number of workers to use when rendering images in parallel.")

_VERBS = frozenset("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")


class _FontHelper:
    def __init__(self, font: ttLib.TTFont):
        self.font = font


def make_dataset() -> None:
    """Writes the image files to a specified directory."""

    list_of_kwargs_to_function = []

    for f_idx, font_path in enumerate(_FONT_PATHS.value):
        logging.info("Examining %s", font_path)
        font_helper = font_helper_lib.FontHelper(
            font=ttLib.TTFont(file=font_path, fontNumber=2),
            input_region=region_lib.Region.G)
        decomposer = decomposer_lib.Decomposer()

        n = 0
        for character in font_helper.known_characters():
            try:
                ideographic_description_sequence = decomposer.decompose(
                    font_helper.region, character).decomposition
            except decomposer_lib.NoDecompositionError:
                continue

            # Only think about "⿰.." for now.
            if not re.match(r"⿰..", ideographic_description_sequence):
                continue

            # Expecting exactly three characters
            if len(ideographic_description_sequence) != 3:
                continue

            unused, lhs, rhs = ideographic_description_sequence
            del unused

            if lhs in _VERBS or rhs in _VERBS:
                continue

            if not font_helper.has_ttglyph(lhs) or not font_helper.has_ttglyph(
                    rhs):
                continue

            # Example: "0013334"
            handle = f"{ord(character):07d}"
            # Example: /tmp/output/0013334.png
            out_path = os.path.join(_DEST.value, f"{f_idx}-{handle}.png")

            list_of_kwargs_to_function.append({
                "font_path": font_path,
                "inputs": [lhs, rhs, character],
                "out_path": out_path,
            })

            n += 1

            if _LIMIT.value and n >= _LIMIT.value:
                break

    def write_dataimage_kwargs(args):
        logging.info("Rendering %s with %s", ''.join(args['inputs']),
                     os.path.split(args['font_path'])[-1])
        return imageutils.write_dataimage(**args)

    with ThreadPool(processes=_NUM_WORKERS.value) as pool:
        return pool.map(write_dataimage_kwargs, list_of_kwargs_to_function)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    if _DEST.value is None:
        raise app.UsageError("Must specify --dest.")
    if _FONT_PATHS.value is []:
        raise app.UsageError("Must specify --fonts.")

    make_dataset()


if __name__ == "__main__":
    app.run(main)
