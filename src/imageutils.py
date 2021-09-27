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
"""Image-handling utilities."""

from PIL import Image
from PIL import ImageChops
from PIL import ImageOps
from absl import flags
from typing import List, Text
import imagehash
import subprocess
import tempfile
import tensorflow as tf

_HB_VIEW_PATH = flags.DEFINE_string(
    "hb_view_path", "hb-view",
    "Path to harfbuzz's hb-view util. Assume this is on the PATH by default.")


def write_character(font_path: Text, character: Text, out_path: Text):
    """Given a character, writes it to a local textfile.

  Args:
    font_path: The path to the font to use.
    character: The character to write.
    out_path: The filename of the image of the character.
  """
    font_size = 256
    margin = "-73,0,-50,0" if "Sans" in font_path else "-70,0,-42,0"

    subprocess.run(
        [
            # Path to the harfbuzz util.
            _HB_VIEW_PATH.value,
            #
            # Path to the font.
            font_path,
            #
            # Path to the output image file.
            f"--output-file={out_path}",
            #
            # Hard-coded to match the GAN architecture in model.py.
            f"--font-size={font_size}",
            #
            # Experimentally derived for a 256x256 image. Other characters might
            # experience clipping, but CJK characters are square and don't
            # suffer from this small margin.
            #
            f"--margin={margin}",
            #
            # The character itself.
            character,
        ],
        check=True)
    # Resize the output image to 256x256.
    subprocess.run([
        "mogrify",
        "-resize",
        "256x256",
        "-gravity",
        "center",
        "-extent",
        "256x256",
        out_path,
    ],
        check=True)


def horizontal_stitch(file_paths: List[Text], out_path: Text):
    # On MacOS, run `brew install imagemagick`.
    # convert image1.png image2.png image3.png +append /tmp/123.png
    subprocess.run(["convert"] + file_paths + ["+append", out_path],
                   check=True)


def write_dataimage(font_path: Text, inputs: List[Text], out_path: Text):
    """Given some characters, renders each as an image and then concats the imgs.

  Args:
    font_path: The path to the font to use.
    inputs: The text inputs. Example: ['亻', '尔']
    out_path: The path to write the resultant, concatenated image to. Example:
      /tmp/1234.png.
  """
    # Write [a], [b], and [ab] to files.
    tmpfiles = []
    for item in inputs:
        tmpfile = tempfile.NamedTemporaryFile(suffix=".png")
        write_character(font_path, item, tmpfile.name)
        tmpfiles.append(tmpfile)

    # Horizontally stitch the images from the inputs.
    horizontal_stitch(file_paths=[tmpfile.name for tmpfile in tmpfiles],
                      out_path=out_path)


def predict_from_paths(generator, input_paths: List[Text], out_path: Text):
    """Given a generator, font and components, renders predicted chr to out_path.

  Args:
    generator: A keras.Model to use to generate characters.
    input_paths: The paths to the 256x256 images of the components
    lhs_path: The path to the 256x256 image of the LHS component.
    rhs_path: The path to the 256x256 image of the RHS component.
    out_path: The path to write the rendered image.
  """

    # Load those images in again.
    images = [
        tf.image.decode_png(tf.io.read_file(path), channels=1)
        for path in input_paths
    ]  # greyscale

    # Create tf.Tensor[256,256,1] tensors of the LHS and RHS images.
    images = [tf.cast(image, tf.float32) for image in images]

    # Create a tf.Tensor[256,256,2] tensor of the LHS and RHS images
    # superimposed.
    inputs = tf.concat(images, axis=2)

    # ...and predict an image.
    # NB: I don't quite understand why, but setting |training=False| here results
    # in a predicted image full of NaN values.
    output = generator(inputs[tf.newaxis, ...], training=True)

    # Flush the predicted image to the filesystem.
    tf.keras.preprocessing.image.array_to_img(output[0]).save(out_path)


def predict(generator, font_path: Text, inputs: List[Text], out_path: Text):
    """Given a generator, font and components, renders predicted chr.

  to out_path.

  Args:
    generator: A keras.Model to use to generate characters.
    font_path: The path to the font from which to pull images of components.
    inputs: The components. In 你, [亻, 尔].
    out_path: The path to write the rendered image.
  """
    # Create a local tempfile to write to, and write the characters to it.
    tmpfiles = [
        tempfile.NamedTemporaryFile(suffix=".png") for i in range(len(inputs))
    ]
    for idx, input in enumerate(inputs):
        write_character(font_path, input, tmpfiles[idx].name)
    predict_from_paths(generator, [tmpfile.name for tmpfile in tmpfiles],
                       out_path)
    for tmpfile in tmpfiles:
        del tmpfile


def superimpose(a_path: Text, b_path: Text, out_path: Text):
    """Given two image paths, writes their superposition to out_path.

  Both images should be greyscale; they will be convered to red and blue
  respectively.

  Args:
    a_path: Path to one image.
    b_path: Path to another image.
    out_path: The path to which to write the new image.
  """
    im1 = Image.open(a_path).convert("L")
    im2 = Image.open(b_path).convert("L")
    im1 = ImageOps.colorize(im1, black="red", white="white")
    im2 = ImageOps.colorize(im2, black="blue", white="white")
    ImageChops.darker(im1, im2).save(out_path)


def wrongness(a_path: Text, b_path: Text) -> float:
    """Given two image paths, calculates their % difference.

  Args:
    a_path: Path to one image.
    b_path: Path to another image.

  Returns:
    Their % difference.
  """
    hash_size = 256
    difference = imagehash.average_hash(
        Image.open(a_path), hash_size=hash_size) - imagehash.average_hash(
            Image.open(b_path), hash_size=hash_size)
    return difference / (hash_size**2)
