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

from PIL import Image
from tensorflow.python.platform import googletest
import imagehash
import os
import tempfile

from src.tools import make_dataset_lib

_FONT_PATHS = ["src/testutils/NotoSerifCJKsc-Regular.otf"]


class TestMakeDataset(googletest.TestCase):
    def test_make_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            make_dataset_lib.make_dataset(dest=tmp_dir,
                                          font_paths=_FONT_PATHS,
                                          split_style=1,
                                          num_workers=1,
                                          limit=1)
            im_actual = Image.open(os.path.join(tmp_dir, "0-0013334.png"))
            im_expected = Image.open(
                "src/testutils/corpus/split-1/0-0013334.png")
            self.assertEqual(
                0,
                imagehash.average_hash(im_actual) -
                imagehash.average_hash(im_expected))

    def test_make_dataset_2(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            make_dataset_lib.make_dataset(dest=tmp_dir,
                                          font_paths=_FONT_PATHS,
                                          split_style=2,
                                          num_workers=1,
                                          limit=1)
            im_actual = Image.open(os.path.join(tmp_dir, "0-0013312.png"))
            im_expected = Image.open(
                "src/testutils/corpus/split-2/0-0013312.png")
            self.assertEqual(
                0,
                imagehash.average_hash(im_actual) -
                imagehash.average_hash(im_expected))


if __name__ == "__main__":
    googletest.main()
