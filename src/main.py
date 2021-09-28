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
"""For usage, see README.md."""

from absl import app
from absl import logging
from absl import flags
from src import imageutils
from src.utils import decomposer as decomposer_lib
from src.utils import region as region_lib
from models import generator_lib
from typing import Sequence
import os
import tensorflow as tf

_INPUT_HELP = "If --input is a single unicode character, we perform "
"decomposition and attempt to render it. Otherwise, we expect a "
"well-formed sequence like "
"'⿰女子' or '⿱占灬' where the first characters is a verb from the "
"following set: ⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻."
_FONT_PATH = flags.DEFINE_string("font_path", None, "Font file to use.")
_INPUT = flags.DEFINE_string("input", "", _INPUT_HELP)
_OUT = flags.DEFINE_string("out", "/tmp/out.png",
                           "Path to write the predicted image.")

_VERBS = frozenset("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")
_GENERATOR_BY_VERB = {
    "⿰": generator_lib.PATH_TO_GENERATOR_SPLIT_1,
    "⿱": generator_lib.PATH_TO_GENERATOR_SPLIT_2,
}


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    if not _FONT_PATH.value:
        raise app.UsageError("Must provide a --font_path.")

    input = _INPUT.value
    if len(input) == 0:
        raise app.UsageError("--inputs must be set.")

    verb = None
    nouns = None
    if len(input) == 1:
        decomposition = decomposer_lib.Decomposer().decompose(
            input_region=region_lib.Region.G, character=input[0]).decomposition
        if decomposition == input:
            raise app.UsageError(
                "Cannot decompose and render a character without a decomposition. See ids.txt."
            )
        verb = decomposition[0]
        nouns = decomposition[1:]
    elif len(input) == 2:
        raise app.UsageError("--input must not be of length two. " +
                             _INPUT_HELP)
    elif input[0] not in _VERBS:
        raise app.UsageError(
            "If --input is an ideographic description "
            "sequence, the first character must be a verb from the following "
            "set: ⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻.")
    else:
        verb = input[0]
        nouns = input[1:]

    logging.info(f"Running on components: {nouns}")
    logging.info(f"Using verb: {verb}")

    if verb not in _GENERATOR_BY_VERB.keys():
        raise app.UsageError(
            f"Please use a supported verb from {_GENERATOR_BY_VERB.keys()}. "
            "See https://github.com/google/autocjk/issues/22.")

    generator_path = _GENERATOR_BY_VERB[verb]
    logging.info(f"Using generator at path: {generator_path}")
    generator = tf.keras.models.load_model(generator_path)

    imageutils.predict(generator, _FONT_PATH.value, nouns, _OUT.value)


if __name__ == "__main__":
    app.run(main)
