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

from absl import logging
from fontTools import ttLib
from multiprocessing.pool import ThreadPool
from typing import Text
import os
import re
from typing import Text

from src import imageutils as imageutils_lib
from src.utils import font_helper as font_helper_lib
from src.utils import decomposer as decomposer_lib
from src.utils import region as region_lib
from src.utils import shape as shape_lib

_VERBS = frozenset("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")


def _make_regex(verb: Text, num_subshapes: int) -> Text:
    # example: ⿰ => r"⿰.."
    return verb + "." * num_subshapes


def _write_dataimage_kwargs(args):
    logging.info("Rendering %s with %s", ''.join(args['inputs']),
                 os.path.split(args['font_path'])[-1])
    return imageutils_lib.write_dataimage(**args)


def validator(font_helper, ids, regex, num_subshapes):
    # Checks that an IDS is "valid", i.e. matches the regex, has the right
    # number of characters, has the right characters, etc.
    if not re.match(regex, ids):
        return False

    if len(ids) != (num_subshapes + 1):
        return False

    if any(i in _VERBS for i in ids[1:]):
        return False

    if any(not font_helper.has_ttglyph(i) for i in ids[1:]):
        return False

    return True


def make_dataset(dest,
                 font_paths,
                 split_style: int,
                 num_workers: int = 40,
                 limit: int = None) -> None:
    """Writes the image files to a specified directory."""

    list_of_kwargs_to_function = []

    verb = sorted(list(_VERBS))[split_style - 1]
    num_subshapes = shape_lib.num_subshapes_of_verb(verb)
    logging.info("Only considering verb %d(%s); expecting %d subshapes",
                 split_style, verb, num_subshapes)
    regex = _make_regex(verb, num_subshapes)

    decomposer = decomposer_lib.Decomposer()

    for f_idx, font_path in enumerate(font_paths):
        logging.info("Examining %s", font_path)
        font_helper = font_helper_lib.FontHelper(
            font=ttLib.TTFont(file=font_path, fontNumber=2),
            input_region=region_lib.Region.G)

        n = 0
        for character in font_helper.known_characters():
            try:
                ids = decomposer.decompose(font_helper.region,
                                           character).decomposition
            except decomposer_lib.NoDecompositionError:
                continue

            if not validator(font_helper, ids, regex, num_subshapes):
                continue

            # Example: "0013334"
            handle = f"{ord(character):07d}"
            # Example: /tmp/output/0013334.png
            out_path = os.path.join(dest, f"{f_idx}-{handle}.png")

            list_of_kwargs_to_function.append({
                "font_path": font_path,
                "inputs": ids[1:] + character,
                "out_path": out_path,
            })

            n += 1

            if limit and n >= limit:
                break

    with ThreadPool(processes=num_workers) as pool:
        return pool.map(_write_dataimage_kwargs, list_of_kwargs_to_function)
