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
"""For evaluating performance against GlyphWiki characters.

Usage:
```
bazel run //src:evaluate_glyphwiki -- \
    --alsologtostderr \
    --font_path=<path_to_reference_font> \
    --limit=10 \
    --out_dir=/tmp/output
```

How do we test model performance against a "new" character, i.e. one not
rendered in most CJK fonts?

Glyphwiki (glyphwiki.org) has PNG and SVG renderings for most characters, even
rare ones. Take U+2cf56 (𬽖) for example. Since it lies in  Unified Ideographs
Extension F, most fonts don't support it yet. It is  represented as ⿰市來.

You can visit https://glyphwiki.org/wiki/u2cf56 or
https://glyphwiki.org/glyph/u2cf56.png for to see a 200x200 PNG of the
character.

So we download the glyphwiki-style PNGs for '市' and '來', feed them as input to
our model, and expect a PNG which looks like the glyphwiki-style PNG for
U+2cf56.

Repeat for any/all characters not supported in a local CJK font.
"""

from PIL import Image
from absl import app
from absl import flags
from absl import logging
from fontTools import ttLib
from models import generator_lib
from multiprocessing.pool import ThreadPool
from src import imageutils
from src.utils import decomposer as decomposer_lib
from src.utils import font_helper as font_helper_lib
from src.utils import region as region_lib
from typing import Iterable, Optional, Sequence, Text, Tuple
import functools
import os
import random
import re
import requests
import statistics
import tempfile
import tensorflow as tf

_VERBS = frozenset("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")

_FONT_PATH = flags.DEFINE_string(
    "font_path", None, "Font file against which to select absent characters.")
_LIMIT = flags.DEFINE_integer(
    "limit", None, "If set, the max number of characters to evaluate.")
_NUM_WORKERS = flags.DEFINE_integer(
    "num_workers", 16, "The number of workers to use in parallel.")
_OUT_DIR = flags.DEFINE_string(
    "out_dir", None,
    "If set, writes [a | b | real(a,b) | predicted(a,b)] images to this directory."
)


@functools.lru_cache(maxsize=7000)
def _download_from_glyphwiki(character: Text, out_file: Text):
    """Downloads an image from glyphwiki as a 200x200 png."""
    # character = "你"
    # ord(..) = 20320
    # hex(..) = 0x4f60
    # str(..).split('x')[1] = '4f60'
    handle = str(hex(ord(character))).split("x")[1]  # '4f60', for example.

    response = requests.get(f"https://glyphwiki.org/glyph/u{handle}.png")

    file = open(out_file, "wb")
    file.write(response.content)
    file.close()

    im1 = Image.open(out_file)
    im1 = im1.resize((256, 256))
    im1.save(out_file, "PNG")


def _get_viable_ids(font_path: Text) -> Iterable[Tuple[Text, Text, Text]]:
    """Yields an iterable of [(亻, 尔, 你), ...]..

  Args:
    font_path: The path to the font file.

  Yields:
    An iterable of pairs of (lhs, rhs, character).
  """
    font_helper = font_helper_lib.FontHelper(font=ttLib.TTFont(file=font_path,
                                                               fontNumber=2),
                                             input_region=region_lib.Region.G)
    decomposer = decomposer_lib.Decomposer()

    # NB: UNKNOWN characters.
    unknown_characters = list(font_helper.unknown_characters())
    random.shuffle(unknown_characters)
    for character in unknown_characters:
        try:
            ids = decomposer.decompose(font_helper.region,
                                       character).decomposition
        except decomposer_lib.NoDecompositionError:
            continue

        # Only think about "⿰.." for now.
        if not re.match(r"⿰..", ids):
            continue

        lhs, rhs = ids[1], ids[2]

        if lhs in _VERBS or rhs in _VERBS:
            continue

        yield (lhs, rhs, character)


def calculate_wrongness(generator,
                        lhs: Text,
                        rhs: Text,
                        character: Text,
                        out_dir: Optional[Text] = None) -> Tuple[Text, float]:
    """For a character, generates the predicted image and returns Wrongness.

  Args:
    generator: a keras.Model which can generate characters.
    lhs: A string, the left-hand component.
    rhs: A string, the right-hand component.
    character: A string, the target character.
    out_dir: If set, an image of the form [a | b | real(a,b) | predicted(a,b)]
      is written to this directory.

  Returns:
    A tuple of the input character and a wrongness %.

  """
    with tempfile.NamedTemporaryFile(
            delete=False, suffix=".png") as predicted_file, open(
                os.path.join(out_dir, f"0x{ord(character):d}.png"), "wb"
    ) as out_file, tempfile.NamedTemporaryFile(
                suffix=".png"
    ) as superimposed_file, tempfile.NamedTemporaryFile(
                suffix=".png") as lhs_file, tempfile.NamedTemporaryFile(
                    suffix=".png") as rhs_file, tempfile.NamedTemporaryFile(
                        suffix=".png") as actual_file:
        _download_from_glyphwiki(lhs, lhs_file.name)
        _download_from_glyphwiki(rhs, rhs_file.name)
        imageutils.predict_from_paths(generator,
                                      [lhs_file.name, rhs_file.name],
                                      predicted_file.name)
        _download_from_glyphwiki(character, actual_file.name)

        if out_dir:
            imageutils.superimpose(actual_file.name, predicted_file.name,
                                   superimposed_file.name)

            imageutils.horizontal_stitch([
                lhs_file.name, rhs_file.name, actual_file.name,
                predicted_file.name, superimposed_file.name
            ], out_file.name)

        return character, imageutils.wrongness(predicted_file.name,
                                               actual_file.name)


def _calculate_wrongness_kwargs(args):
    logging.info("Rendering %s %s %s to %s", args['lhs'], args['rhs'],
                 args['character'], args['out_dir'])
    return calculate_wrongness(**args)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    if not _FONT_PATH.value:
        raise app.UsageError(
            "Must specify a reference font; i.e., glyphs absent from this font will be evaluated."
        )

    # Load in the generator,
    generator = tf.keras.models.load_model(generator_lib.PATH_TO_GENERATOR)

    n = 0

    list_of_kwargs_to_function = []
    for lhs, rhs, character in _get_viable_ids(_FONT_PATH.value):
        if _LIMIT.value and n >= _LIMIT.value:
            break
        n = n + 1
        list_of_kwargs_to_function.append({
            "generator": generator,
            "lhs": lhs,
            "rhs": rhs,
            "character": character,
            "out_dir": _OUT_DIR.value,
        })

    pool = ThreadPool(processes=_NUM_WORKERS.value)
    result = pool.map(_calculate_wrongness_kwargs, list_of_kwargs_to_function)
    pool.close()
    pool.join()

    # map from character to wrongness.
    wrongness_map = {}
    for c, w in result:
        wrongness_map[c] = w

    logging.info("\tMean: %.3f", statistics.mean(wrongness_map.values()))
    logging.info("\tVariance: %.3f",
                 statistics.variance(wrongness_map.values()))
    logging.info("\tLeast wrongness: %s",
                 min(wrongness_map, key=wrongness_map.get))
    logging.info("\tMost wrongness: %s",
                 max(wrongness_map, key=wrongness_map.get))


if __name__ == "__main__":
    app.run(main)
