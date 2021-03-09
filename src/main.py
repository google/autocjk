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
from models import generator_lib
from typing import Sequence
import os
import tensorflow as tf

_FONT_PATH = flags.DEFINE_string("font_path", None, "Font file to use.")
_LHS = flags.DEFINE_string("lhs", None, "Left-hand character.")
_RHS = flags.DEFINE_string("rhs", None, "Right-hand character.")
_OUT = flags.DEFINE_string("out", "/tmp/out.png",
                           "Path to write the predicted image.")


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    if not _LHS.value:
        raise app.UsageError("Must provide a --lhs.")
    if not _RHS.value:
        raise app.UsageError("Must provide a --rhs.")
    if not _FONT_PATH.value:
        raise app.UsageError("Must provide a --font_path.")

    # Load in the generator,
    generator = tf.keras.models.load_model(generator_lib.PATH_TO_GENERATOR)
    imageutils.predict(generator, _FONT_PATH.value, _LHS.value, _RHS.value,
                       _OUT.value)


if __name__ == "__main__":
    app.run(main)
