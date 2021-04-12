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
_PATH_TO_EXPECTED_2CF56 = "src/testutils/predicted-2CF56.png"
_PATH_TO_EXPECTED_653E = "src/testutils/predicted-653E.png"

# There's a bit of variance between runs. A value of |8| below
# passes with --runs_per_test=1000 --runs_per_test_detects_flakes.
_EXPECTED_VARIANCE = 8


# Run main.py on Serif SC Regular, predict one character, and compare it
# to a golden file.
class TestMain(googletest.TestCase):
    def test_successful_run_lhs_and_rhs(self):
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".png") as tmp_file:
            self.assertCommandSucceeds(command=[
                'src/main', '--font_path', os.path.abspath(_PATH_TO_FONT),
                '--lhs=市', '--rhs=來', '--out', tmp_file.name,
            ], env={'PATH': os.environ['PATH'], })

            image_actual = Image.open(tmp_file.name)
            image_expected = Image.open(_PATH_TO_EXPECTED_2CF56)

            hash_actual = imagehash.average_hash(image_actual)
            hash_expected = imagehash.average_hash(image_expected)

            self.assertLessEqual(
                hash_actual - hash_expected, _EXPECTED_VARIANCE)

    def test_successful_run_chr(self):
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".png") as tmp_file:
            self.assertCommandSucceeds(command=[
                'src/main', '--font_path', os.path.abspath(_PATH_TO_FONT),
                '--chr=放', '--out', tmp_file.name,
            ], env={'PATH': os.environ['PATH'], })

            image_actual = Image.open(tmp_file.name)
            image_expected = Image.open(_PATH_TO_EXPECTED_653E)

            hash_actual = imagehash.average_hash(image_actual)
            hash_expected = imagehash.average_hash(image_expected)

            self.assertLessEqual(
                hash_actual - hash_expected, _EXPECTED_VARIANCE)

    def test_no_font_path(self):
        self.assertCommandFails(
            command=[
                'src/main',
                '--lhs=市',
                '--rhs=來'],
            env={
                'PATH': os.environ['PATH'],
            },
            regexes=[".*Must provide a --font_path.*"])

    def test_no_rhs(self):
        self.assertCommandFails(
            command=[
                'src/main',
                '--font_path',
                os.path.abspath(_PATH_TO_FONT),
                '--lhs=来'],
            env={
                'PATH': os.environ['PATH'],
            },
            regexes=[".*Either --chr must be set, or both --lhs and --rhs must be set..*"])

    def test_no_lhs(self):
        self.assertCommandFails(
            command=[
                'src/main',
                '--font_path',
                os.path.abspath(_PATH_TO_FONT),
                '--rhs=来'],
            env={
                'PATH': os.environ['PATH'],
            },
            regexes=[".*Either --chr must be set, or both --lhs and --rhs must be set..*"])

    def test_invalid_chr(self):
        self.assertCommandFails(
            command=[
                'src/main',
                '--font_path',
                os.path.abspath(_PATH_TO_FONT),
                '--chr=来'],
            env={
                'PATH': os.environ['PATH'],
            },
            regexes=[".*If providing --chr, must provide a character which decomposes to.*"])


if __name__ == "__main__":
    googletest.main()
