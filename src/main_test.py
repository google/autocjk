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
from typing import Text
import imagehash
import os
import subprocess
import tempfile

_PATH_TO_FONT = os.path.abspath("src/testutils/NotoSerifCJKsc-Regular.otf")
_PATH_TO_EXPECTED_2CF56 = "src/testutils/predicted-2CF56.png"
_PATH_TO_EXPECTED_653E = "src/testutils/predicted-653E.png"
_PATH_TO_EXPECTED_70B9 = "src/testutils/predicted-70B9.png"

# There's a bit of variance between runs. The value below
# passes with --runs_per_test=1000 --runs_per_test_detects_flakes.
_EXPECTED_VARIANCE = 10


def imagehash_difference_by_path(path1: Text, path2: Text) -> int:
    image1 = Image.open(path1)
    image2 = Image.open(path2)

    hash1 = imagehash.average_hash(image1)
    hash2 = imagehash.average_hash(image2)
    return hash1 - hash2


# Run main.py on Serif SC Regular, predict one character, and compare it
# to a golden file.
class TestMain(googletest.TestCase):
    def test_successful_run_inputs_split_1(self):
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".png") as tmp_file:
            self.assertCommandSucceeds(command=[
                'src/main',
                f'--font_path={_PATH_TO_FONT}',
                '--input=⿰市來',
                f'--out={ tmp_file.name}',
            ],
                env={'PATH': os.environ['PATH']})

            self.assertLessEqual(
                imagehash_difference_by_path(tmp_file.name,
                                             _PATH_TO_EXPECTED_2CF56),
                _EXPECTED_VARIANCE)

    def test_successful_run_inputs_single_character_decomposition(self):
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".png") as tmp_file:
            self.assertCommandSucceeds(command=[
                'src/main',
                f'--font_path={_PATH_TO_FONT}',
                '--input=放',
                f'--out={tmp_file.name}',
            ],
                env={'PATH': os.environ['PATH']})

            self.assertLessEqual(
                imagehash_difference_by_path(tmp_file.name,
                                             _PATH_TO_EXPECTED_653E),
                _EXPECTED_VARIANCE)

    def test_failed_run_inputs_split_1(self):
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".png") as tmp_file:
            self.assertCommandFails(
                command=[
                    'src/main',
                    f'--font_path={_PATH_TO_FONT}',
                    '--input=占灬',  # One must specify the verb here.
                    f'--out={tmp_file.name}',
                ],
                env={'PATH': os.environ['PATH']},
                regexes=[".*input must not be of length two.*"])

    def test_failed_run_inputs_split_2(self):
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".png") as tmp_file:
            self.assertCommandSucceeds(
                command=[
                    'src/main',
                    f'--font_path={_PATH_TO_FONT}',
                    '--input=⿱占灬',
                    f'--out={tmp_file.name}',
                ],
                env={'PATH': os.environ['PATH']})

            self.assertLessEqual(
                imagehash_difference_by_path(tmp_file.name,
                                             _PATH_TO_EXPECTED_70B9),
                _EXPECTED_VARIANCE)

    def test_no_font_path(self):
        self.assertCommandFails(command=['src/main', '--input=市'],
                                env={
                                    'PATH': os.environ['PATH'],
        },
            regexes=[".*Must provide a --font_path.*"])

    def test_invalid_character(self):
        self.assertCommandFails(
            command=[
                'src/main',
                f'--font_path={_PATH_TO_FONT}',
                '--input=来',
            ],
            env={
                'PATH': os.environ['PATH'],
            },
            regexes=[
                ".*Cannot decompose and render a character without a decomposition. See ids.txt..*"
            ])


if __name__ == "__main__":
    googletest.main()
