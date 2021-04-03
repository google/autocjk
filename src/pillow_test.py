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
import tempfile


class TestClass(googletest.TestCase):
    def test(self):
        # Just need to make sure `from PIL import Image` works.
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".png") as tmp_file:
            Image.new("RGBA", (64, 64)).save(tmp_file.name)
            print(Image.open(tmp_file.name))


if __name__ == "__main__":
    googletest.main()
