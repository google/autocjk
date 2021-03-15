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
import subprocess
import tempfile


# Run make_dataset.py on Serif SC Regular, generate one image, and compare it
# to a golden file.
class TestMakeDataset(googletest.TestCase):
    def test_no_dest(self):
        self.assertCommandFails(command=['src/make_dataset'],
                                regexes=[".*Must specify --dest.*"])

    def test_no_fonts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertCommandFails(
                command=['src/make_dataset', '--dst', tmp_dir],
                regexes=[".*Must specify --fonts.*"])

    def test_make_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertCommandSucceeds(command=[
                "src/make_dataset", "--dst", tmp_dir, "--fonts",
                "src/testutils/NotoSerifCJKsc-Regular.otf", "--limit=1",
                "--num_workers=1"
            ])
            im_actual = Image.open(os.path.join(tmp_dir, "0-0013334.png"))
            im_expected = Image.open("src/testutils/0-0013334.png")
            self.assertEqual(
                0,
                imagehash.average_hash(im_actual) -
                imagehash.average_hash(im_expected))


if __name__ == "__main__":
    googletest.main()
