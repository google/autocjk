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

from tensorflow.python.platform import googletest
import os
import tempfile


class TestMakeDataset(googletest.TestCase):
    def test_no_dest(self):
        self.assertCommandFails(command=['src/tools/make_dataset'],
                                env={
                                    'PATH': os.environ['PATH'],
        },
            regexes=[".*Must specify --dest.*"])

    def test_no_fonts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertCommandFails(
                command=['src/tools/make_dataset', '--dst', tmp_dir],
                env={
                    'PATH': os.environ['PATH'],
                },
                regexes=[".*Must specify --fonts.*"])


if __name__ == "__main__":
    googletest.main()
