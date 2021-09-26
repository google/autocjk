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
_PATH_TO_TRAINING_IMAGE = "src/testutils/corpus/split-1/0-0013334.png"
_PATH_TO_TRAINING_IMAGE_WRONG_SIZE = "src/testutils/training_data/0-0013312-250.png"


class TestClass(googletest.TestCase):
    def test_load_image(self):
        # the input image is [ a | b | a+b ]; we expect each segment to be
        # 256x256.
        source = Image.open(_PATH_TO_TRAINING_IMAGE)
        self.assertEqual(source.size, (256 * 3, 256))
        # assert that the model_lib._load_image() util correctly splits this
        # into three 256x256 segments.
        a, b, a_and_b = model_lib._load_image(_PATH_TO_TRAINING_IMAGE)
        self.assertEqual(a.shape, (256, 256, 1))
        self.assertEqual(b.shape, (256, 256, 1))
        self.assertEqual(a_and_b.shape, (256, 256, 1))

    def test_load_image_wrong_size(self):
        # This source image is too short,
        source = Image.open(_PATH_TO_TRAINING_IMAGE_WRONG_SIZE)
        self.assertEqual(source.size, (256 * 3, 250))
        # ...but we expect _load_image() to apply
        # tf.image.resize_with_crop_or_pad() internally.
        a, b, a_and_b = model_lib._load_image(_PATH_TO_TRAINING_IMAGE)
        self.assertEqual(a.shape, (256, 256, 1))
        self.assertEqual(b.shape, (256, 256, 1))
        self.assertEqual(a_and_b.shape, (256, 256, 1))


class TrainTestClass(googletest.TestCase):
    def test_train_one_epoch(self):
        train_dataset, test_dataset = model_lib.make_datasets(
            'src/testutils/corpus/split-1/0-*.png')

        generator = model_lib.make_generator()
        discriminator = model_lib.make_discriminator()
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        epochs = 1
        model_lib.fit(generator, generator_optimizer, discriminator,
                      discriminator_optimizer, loss_object, train_dataset,
                      epochs, test_dataset)

        self.assertEqual(generator.input_shape, (None, 256, 256, 2))
        self.assertEqual(generator.output_shape, (None, 256, 256, 1))
        self.assertFalse(generator.run_eagerly)

        with tempfile.NamedTemporaryFile(mode='w+', suffix=".png") as tmp_file:
            # We can't assert anything about the generated image, but we just
            # want to see that imageutils.predict(generator, ...) doesn't fail.
            imageutils.predict(generator, _PATH_TO_FONT, '市', '來',
                               tmp_file.name)


if __name__ == "__main__":
    googletest.main()
