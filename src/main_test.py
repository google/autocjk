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

_PATH_TO_FONT = "src/testutils/NotoSerifCJKsc-Regular.otf"
_PATH_TO_EXPECTED = "src/testutils/predicted-175276.png"

# There's a bit of variance between runs. A value of |8| below
# passes with --runs_per_test=1000 --runs_per_test_detects_flakes.
_EXPECTED_VARIANCE = 8


# Run main.py on Serif SC Regular, predict one character, and compare it
# to a golden file.
class TestMain(googletest.TestCase):
    def test_main(self):
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".png") as tmp_file:
            subprocess.check_output([
                "src/main", "--font_path",
                os.path.abspath(_PATH_TO_FONT), "--lhs=市", "--rhs=來", "--out",
                tmp_file.name
            ])
            im_actual = Image.open(tmp_file.name)
            im_expected = Image.open(_PATH_TO_EXPECTED)
            self.assertLessEqual(
                imagehash.average_hash(im_actual) -
                imagehash.average_hash(im_expected), _EXPECTED_VARIANCE)


if __name__ == "__main__":
    googletest.main()
