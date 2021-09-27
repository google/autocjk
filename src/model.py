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
"""GAN for generating CJK characters.

The vast majority of this code is adapted from the pix2pix GAN described in
https://www.tensorflow.org/tutorials/generative/pix2pix. Changes include the
specific tensor dimensions, some tuning of magic numbers, and some changes to
loss functions.

TODO(ambuc): This file has type annotations because they're useful for a human
reader, but the build system doesn't yet enforce them with a strictly-typed
python build rule.
"""

import time
from absl import logging
from typing import List, Text, Tuple, Optional

from IPython import display
import matplotlib.pyplot as plt
import tensorflow as tf


def _load_image(filename: Text, num_cells: int) -> List[List[tf.Tensor]]:
    """Given the filename of a PNG, returns a list of three tensors: a, b, a+b.

  Args:
    filename: Path to a file. The file must be a PNG and greyscale and 256x256.
    num_cells: the number of square 256x256 cells to expect per input image.

  Returns:
    A list of tensors: a, b, and a+b.

  """
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)  # greyscale
    image = tf.cast(image, tf.float32)
    return [
        tf.image.crop_to_bounding_box(image, 0, i * 256, 256, 256)
        for i in range(num_cells)
    ]


def make_datasets(files_glob: Text,
                  num_cells: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Makes the train/test datasets.

  Args:
    files_glob: A glob (like "/tmp/folder/*.png") of all the input images.
    num_cells: the number of square 256x256 cells to expect per input image.

  Returns:
    A pair of train, test datasets of type tf.data.Dataset.

  """
    ds = tf.data.Dataset.list_files(files_glob).map(
        lambda i: _load_image(i, num_cells),
        num_parallel_calls=tf.data.AUTOTUNE).shuffle(400).batch(1)

    train_dataset_a = ds.shard(num_shards=3, index=0)
    train_dataset_b = ds.shard(num_shards=3, index=1)
    train_ds = train_dataset_a.concatenate(train_dataset_b)

    test_ds = ds.shard(num_shards=3, index=2)

    return train_ds, test_ds


def _downsample(filters: int,
                size: int,
                apply_batchnorm: bool = True) -> tf.keras.Sequential:
    """Downsampler from https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_generator.

  Args:
    filters: The number of filters.
    size: The size of the input tensor width at this step.
    apply_batchnorm: Whether or not to apply batch normalization. Probably
      should be false on the input layer, and true elsewhere.

  Returns:
    A sequential model.
  """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters,
                               size,
                               strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def _upsample(filters: int,
              size: int,
              apply_dropout: bool = False) -> tf.keras.Sequential:
    """Upsampler from https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_generator.

  Args:
    filters: The number of filters.
    size: The size of the input tensor width at this step.
    apply_dropout: Whether or not to apply dropout. Probably should be true for
      the first few layers and false elsewhere.

  Returns:
    A sequential model.

  """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters,
                                        size,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def generate_images(model: tf.keras.Model, inputs: List[tf.Tensor],
                    target: tf.Tensor) -> None:
    """In Colab, prints [ inputs... | real(a,b) | predicted(a,b)] to the display.

  Args:
    model: The generator to use.
    inputs: the input image(s)
    target: The real(a,b) composition.
  """
    x = tf.concat(inputs, axis=3)
    x = tf.reshape(x, [256, 256, len(inputs)])
    prediction = model(x[tf.newaxis, ...], training=True)

    images = [i[0] for i in inputs] + [target[0], prediction[0]]
    # n inputs + ground truth + prediction
    num_subplots = len(inputs) + 2
    fig, axes = plt.subplots(1, num_subplots)
    titles = [f"Input Image #{i}" for i in range(len(inputs))
              ] + ['Ground Truth', 'Predicted Image']

    for image, axis, title in zip(images, axes, titles):
        axis.set_title(title)
        axis.imshow(image[:, :, 0])
        axis.axis('off')
    fig.show()


class Model():
    def __init__(self,
                 train_ds: tf.data.Dataset,
                 test_ds: tf.data.Dataset,
                 epochs: int,
                 num_cells: int,
                 lambda_arg: int = 100,
                 checkpoint: Optional[tf.train.Checkpoint] = None,
                 checkpoint_prefix: Optional[Text] = None,
                 summary_writer: tf.summary.SummaryWriter = None):
        self._num_cells = num_cells
        self._num_inputs = self._num_cells - 1

        self.generator = self._make_generator()
        self._generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self._discriminator = self._make_discriminator()
        self._discriminator_optimizer = tf.keras.optimizers.Adam(2e-4,
                                                                 beta_1=0.5)
        self._loss_object = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)
        self._train_ds = train_ds
        self._test_ds = test_ds
        self._epochs = epochs
        self._lambda = lambda_arg
        self._checkpoint = checkpoint
        self._checkpoint_prefix = checkpoint_prefix
        self._summary_writer = summary_writer

    def _make_generator(self) -> tf.keras.Model:
        """Creates a generator.

        99% of this is copied directly from
        https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_generator,
        except for the input shape (now two channels, two greyscale images instead of
        one RGB image) and output shape (one channel, one greyscale image instead of
        one RGB image).

        Returns:
          a tf.keras.Model which returns a 256x256x1 tensor.
        """
        inputs = tf.keras.layers.Input(shape=[256, 256, self._num_inputs])

        up_stack = [
            _upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            _upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            _upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            _upsample(512, 4),  # (bs, 16, 16, 1024)
            _upsample(256, 4),  # (bs, 32, 32, 512)
            _upsample(128, 4),  # (bs, 64, 64, 256)
            _upsample(64, 4),  # (bs, 128, 128, 128)
        ]

        x = inputs

        skips = []
        for down in [
                _downsample(64, 4,
                            apply_batchnorm=False),  # (bs, 128, 128, 64)
                _downsample(128, 4),  # (bs, 64, 64, 128)
                _downsample(256, 4),  # (bs, 32, 32, 256)
                _downsample(512, 4),  # (bs, 16, 16, 512)
                _downsample(512, 4),  # (bs, 8, 8, 512)
                _downsample(512, 4),  # (bs, 4, 4, 512)
                _downsample(512, 4),  # (bs, 2, 2, 512)
                _downsample(512, 4),  # (bs, 1, 1, 512)
        ]:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        last = tf.keras.layers.Conv2DTranspose(
            1,  # one output channel, i.e. greyscale
            4,
            strides=2,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            activation='tanh')  # (bs, 256, 256, 3)

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def _make_discriminator(self) -> tf.keras.Model:
        """Returns a discriminator.

        This is 99% the same as
        https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_discriminator,
        except that the shape of the input and output tensors are different.

        Returns:
          A tf.keras.model which accepts a 256x256x2 tensor and compares it to a
          target 256x256x1 tensor.
        """
        initializer = tf.random_normal_initializer(0., 0.02)

        input_img = tf.keras.layers.Input(shape=[256, 256, self._num_inputs],
                                          name='input_image')
        target_img = tf.keras.layers.Input(shape=[256, 256, 1],
                                           name='target_image')

        x = tf.keras.layers.concatenate([input_img, target_img
                                         ])  # (bs, 256, 256, channels*2)

        down1 = _downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
        down2 = _downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = _downsample(256, 4)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512,
                                      4,
                                      strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(
                                          zero_pad1)  # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(
            leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1,
                                      4,
                                      strides=1,
                                      kernel_initializer=initializer)(
                                          zero_pad2)  # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=[input_img, target_img], outputs=last)

    def _generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self._loss_object(tf.ones_like(disc_generated_output),
                                     disc_generated_output)
        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self._lambda * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def _discriminator_loss(self, disc_real_output,
                            disc_generated_output) -> float:
        """Returns discriminator loss.

        100% the same as
        https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_discriminator.

        Args:
          disc_real_output: A set of real images.
          disc_generated_output: A set of generator images.

        Returns:
          The sum of some loss functions.
        """
        real_loss = self._loss_object(tf.ones_like(disc_real_output),
                                      disc_real_output)
        generated_loss = self._loss_object(
            tf.zeros_like(disc_generated_output), disc_generated_output)
        return real_loss + generated_loss

    @tf.function
    def _train_step(self, inputs: List[tf.Tensor], target: tf.Tensor,
                    epoch: int) -> None:
        """Trains the models for one (1) epoch.
      See https://www.tensorflow.org/tutorials/generative/pix2pix#training.
      """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            inp_x = tf.concat(inputs, axis=3)
            gen_output = self.generator(inp_x, training=True)

            disc_real_output = self._discriminator([inp_x, target],
                                                   training=True)
            disc_generated_output = self._discriminator([inp_x, gen_output],
                                                        training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self._generator_loss(
                disc_generated_output, gen_output, target)
            disc_loss = self._discriminator_loss(disc_real_output,
                                                 disc_generated_output)

        # TODO(ambuc): Should this simply be gen_l1_loss?
        generator_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self._discriminator.trainable_variables)

        self._generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        self._discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients,
                self._discriminator.trainable_variables))

        if self._summary_writer:
            with self._summary_writer.as_default():
                tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
                tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
                tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
                tf.summary.scalar('disc_loss', disc_loss, step=epoch)

    def fit(self):
        for epoch in range(self._epochs):
            start = time.time()

            display.clear_output(wait=True)

            for items in self._test_ds.take(1):
                inputs = items[:-1]
                target = items[-1]
                generate_images(self.generator, inputs, target)
            print('Epoch: ', epoch)

            for n, items in self._train_ds.enumerate():
                inputs = items[:-1]
                target = items[-1]
                print('.', end='')
                if (n + 1) % 100 == 0:
                    print()
                self._train_step(inputs, target, epoch)
            print()

            if self._checkpoint and self._checkpoint_prefix:
                self._checkpoint.save(file_prefix=self._checkpoint_prefix)

            print('Time taken for epoch {} is {} sec\n'.format(
                epoch + 1,
                time.time() - start))

        if self._checkpoint and self._checkpoint_prefix:
            self._checkpoint.save(file_prefix=self._checkpoint_prefix)
