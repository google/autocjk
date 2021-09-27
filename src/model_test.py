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
import tensorflow as tf

from src import model as model_lib
from src import imageutils

_PATH_TO_FONT = "src/testutils/NotoSerifCJKsc-Regular.otf"
_PATH_TO_TRAINING_IMAGE_SPLIT_1 = "src/testutils/corpus/split-1/0-0013334.png"
_PATH_TO_TRAINING_IMAGE_SPLIT_3 = "src/testutils/corpus/split-3/0-0014531.png"


class TestClass(googletest.TestCase):
    def test_load_image_split_1(self):
        # the input image is [ a | b | a+b ]; we expect each segment to be
        # 256x256. Assert that the model_lib._load_image() util correctly
        # splits this into three 256x256 segments.
        with Image.open(_PATH_TO_TRAINING_IMAGE_SPLIT_1) as source:
            self.assertEqual(source.size, (256 * 3, 256))
        a, b, sum = model_lib._load_image(_PATH_TO_TRAINING_IMAGE_SPLIT_1,
                                          num_cells=3)
        self.assertEqual(a.shape, (256, 256, 1))
        self.assertEqual(b.shape, (256, 256, 1))
        self.assertEqual(sum.shape, (256, 256, 1))

    def test_load_image_split_3(self):
        # the input image is [ a | b | c | a+b+c ]; we expect each segment to be
        # 256x256. Assert that the model_lib._load_image() util correctly
        # splits this into four 256x256 segments.
        with Image.open(_PATH_TO_TRAINING_IMAGE_SPLIT_3) as source:
            self.assertEqual(source.size, (256 * 4, 256))
        a, b, c, sum = model_lib._load_image(_PATH_TO_TRAINING_IMAGE_SPLIT_3,
                                             num_cells=4)
        self.assertEqual(a.shape, (256, 256, 1))
        self.assertEqual(b.shape, (256, 256, 1))
        self.assertEqual(c.shape, (256, 256, 1))
        self.assertEqual(sum.shape, (256, 256, 1))


class TrainTestClass(googletest.TestCase):
    def test_train_one_epoch(self):
        num_cells = 3
        train_dataset, test_dataset = model_lib.make_datasets(
            'src/testutils/corpus/split-1/0-*.png', num_cells)

        model = model_lib.Model(train_dataset,
                                test_dataset,
                                epochs=1,
                                num_cells=num_cells)
        model.fit()

        self.assertEqual(model.generator.input_shape, (None, 256, 256, 2))
        self.assertEqual(model.generator.output_shape, (None, 256, 256, 1))
        self.assertFalse(model.generator.run_eagerly)

        with tempfile.NamedTemporaryFile(mode='w+', suffix=".png") as tmp_file:
            # We can't assert anything about the generated image, but we just
            # want to see that imageutils.predict(generator, ...) doesn't fail.
            imageutils.predict(model.generator, _PATH_TO_FONT, [
                '市',
                '來',
            ], tmp_file.name)

        # We want to see that model_lib.generate_images(generator, ...) doesn't
        # fail.
        for items in test_dataset.take(1):
            inputs = items[:-1]
            target = items[-1]
            model_lib.generate_images(model.generator, inputs, target)

    def test_train_one_epoch_split_3(self):
        num_cells = 4
        train_dataset, test_dataset = model_lib.make_datasets(
            'src/testutils/corpus/split-3/0-*.png', num_cells)

        model = model_lib.Model(train_dataset,
                                test_dataset,
                                epochs=1,
                                num_cells=num_cells)
        model.fit()

        self.assertEqual(model.generator.input_shape, (None, 256, 256, 3))
        self.assertEqual(model.generator.output_shape, (None, 256, 256, 1))
        self.assertFalse(model.generator.run_eagerly)

        with tempfile.NamedTemporaryFile(mode='w+', suffix=".png") as tmp_file:
            # We can't assert anything about the generated image, but we just
            # want to see that imageutils.predict(generator, ...) doesn't fail.
            imageutils.predict(model.generator, _PATH_TO_FONT, [
                '市',
                '口',
                '來',
            ], tmp_file.name)

        # We want to see that model_lib.generate_images(generator, ...) doesn't
        # fail.
        for items in test_dataset.take(1):
            inputs = items[:-1]
            target = items[-1]
            model_lib.generate_images(model.generator, inputs, target)


if __name__ == "__main__":
    googletest.main()
