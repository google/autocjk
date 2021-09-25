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

Usage:
    mkdir /tmp/corpus
    bazel run //src/tools:make_dataset -- --dst=/tmp/corpus \
        --font_paths=<path_to_font(s)>
"""

from absl import app
from absl import flags
from absl import logging
from typing import Sequence

from src.tools import make_dataset_lib

_DEST = flags.DEFINE_string(
    "dst", None, "Path to a directory into which to write. Required.")
_FONT_PATHS = flags.DEFINE_list("fonts", [], "List of font files.")
_SPLIT_STYLE = flags.DEFINE_integer(
    'split_style', 1,
    'The decomposition style to limit our dataset to.  1=⿰, 2=⿱, 3=⿲, 4=⿳, '
    '5=⿴, 6=⿵, 7=⿶, 8=⿷, 9=⿸, 10=⿹, 11=⿺, 12=⿻')
_NUM_WORKERS = flags.DEFINE_integer(
    "num_workers", 40,
    "Number of workers to use when rendering images in parallel.")
_LIMIT = flags.DEFINE_integer(
    "limit", None,
    "If set, maximum number of training pairs to produce. Useful when testing."
)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    if _DEST.value is None:
        raise app.UsageError("Must specify --dest.")
    if not _FONT_PATHS.value:
        raise app.UsageError("Must specify --fonts.")

    make_dataset_lib.make_dataset(_DEST.value, _FONT_PATHS.value,
                                  _SPLIT_STYLE.value, _NUM_WORKERS.value,
                                  _LIMIT.value)


if __name__ == "__main__":
    app.run(main)
