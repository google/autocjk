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
from absl import flags
from src import imageutils
from src.utils import decomposer as decomposer_lib
from src.utils import region as region_lib
from models import generator_lib
from typing import Sequence
import os
import tensorflow as tf

_FONT_PATH = flags.DEFINE_string("font_path", None, "Font file to use.")
_LHS = flags.DEFINE_string(
    "lhs",
    None,
    "Left-hand character. Either --chr must be set, or both --lhs "
    "and --rhs must be set.")
_RHS = flags.DEFINE_string(
    "rhs",
    None,
    "Right-hand character. Either --chr must be set, or both --lhs "
    "and --rhs must be set.")
_CHR = flags.DEFINE_string(
    "chr",
    None,
    "Character. Either --chr must be set, or both --lhs and --rhs "
    "must be set.")
_OUT = flags.DEFINE_string("out", "/tmp/out.png",
                           "Path to write the predicted image.")


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    if not _CHR.value and not (_LHS.value and _RHS.value):
        raise app.UsageError(
            "Either --chr must be set, or both --lhs and --rhs must be set.")
    if not _FONT_PATH.value:
        raise app.UsageError("Must provide a --font_path.")

    lhs, rhs = None, None
    if _CHR.value:
        decomposition = decomposer_lib.Decomposer().decompose(
            input_region=region_lib.Region.G, character=_CHR.value).decomposition
        if decomposition[0] != "⿰":
            raise app.UsageError(
                "If providing --chr, must provide a character which decomposes "
                "to ⿰XY according to /third_party/IDS.txt. See "
                "https://github.com/google/autocjk/issues/22.")
        lhs, rhs = decomposition[1], decomposition[2]
    else:
        lhs, rhs = _LHS.value, _RHS.value

    # Load in the generator,
    generator = tf.keras.models.load_model(generator_lib.PATH_TO_GENERATOR)
    imageutils.predict(generator, _FONT_PATH.value, lhs, rhs, _OUT.value)


if __name__ == "__main__":
    app.run(main)
