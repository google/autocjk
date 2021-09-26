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
from typing import List, Text, Tuple

from IPython import display
import matplotlib.pyplot as plt
import tensorflow as tf

_LAMBDA = 100


def _load_image(filename: Text) -> List[List[tf.Tensor]]:
    """Given the filename of a PNG, returns a list of three tensors: a, b, a+b.

  Args:
    filename: Path to a file. The file must be a PNG and greyscale and 256x256.

  Returns:
    A list of tensors: a, b, and a+b.

  """
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)  # greyscale

    # Our images have a width which is divisible by three.
    w = tf.shape(image)[1] // 3

    imgs = [
        tf.cast(image[:, n * w:(n + 1) * w, :], tf.float32) for n in range(3)
    ]

    imgs[0] = tf.image.resize_with_crop_or_pad(imgs[0], 256, 256)
    imgs[1] = tf.image.resize_with_crop_or_pad(imgs[1], 256, 256)
    imgs[2] = tf.image.resize_with_crop_or_pad(imgs[2], 256, 256)

    return imgs


def make_datasets(files_glob: Text) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Makes the train/test datasets.

  Args:
    files_glob: A glob (like "/tmp/folder/*.png") of all the input images.

  Returns:
    A pair of train, test datasets of type tf.data.Dataset.

  """
    ds = tf.data.Dataset.list_files(files_glob).map(
        _load_image, num_parallel_calls=tf.data.AUTOTUNE).shuffle(400).batch(1)

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


def make_generator() -> tf.keras.Model:
    """Creates a generator.

  99% of this is copied directly from
  https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_generator,
  except for the input shape (now two channels, two greyscale images instead of
  one RGB image) and output shape (one channel, one greyscale image instead of
  one RGB image).

  Returns:
    a tf.keras.Model which returns a 256x256x1 tensor.
  """
    inputs = tf.keras.layers.Input(shape=[256, 256, 2])

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
            _downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
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


def generator_loss(loss_object: tf.keras.losses.Loss, disc_generated_output,
                   gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output),
                           disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (_LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def make_discriminator() -> tf.keras.Model:
    """Returns a discriminator.

  This is 99% the same as
  https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_discriminator,
  except that the shape of the input and output tensors are different.

  Returns:
    A tf.keras.model which accepts a 256x256x2 tensor and compares it to a
    target 256x256x1 tensor.
  """
    initializer = tf.random_normal_initializer(0., 0.02)

    input_img = tf.keras.layers.Input(shape=[256, 256, 2], name='input_image')
    target_img = tf.keras.layers.Input(shape=[256, 256, 1],
                                       name='target_image')

    x = tf.keras.layers.concatenate([input_img,
                                     target_img])  # (bs, 256, 256, channels*2)

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


def discriminator_loss(loss_object: tf.keras.losses.Loss, disc_real_output,
                       disc_generated_output) -> float:
    """Returns discriminator loss.

  100% the same as
  https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_discriminator.

  Args:
    loss_object: A reusable loss_object of type
      tf.keras.losses.BinaryCrossentropy.
    disc_real_output: A set of real images.
    disc_generated_output: A set of generator images.

  Returns:
    The sum of some loss functions.
  """
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output),
                                 disc_generated_output)

    return real_loss + generated_loss


def generate_images(model: tf.keras.Model, input_a: tf.Tensor,
                    input_b: tf.Tensor, target: tf.Tensor) -> None:
    """In Colab, prints [a | b | real(a,b) | predicted(a,b)] to the display.

  Args:
    model: The generator to use.
    input_a: the LHS image.
    input_b: the RHS image.
    target: The real(a,b) composition.
  """
    x = tf.concat([input_a, input_b], 3)
    x = tf.reshape(x, [256, 256, 2])
    prediction = model(x[tf.newaxis, ...], training=True)

    images = [input_a[0], input_b[0], target[0], prediction[0]]
    fig, axes = plt.subplots(1, 4)
    titles = [
        'Input Image A', 'Input Image B', 'Ground Truth', 'Predicted Image'
    ]

    for image, axis, title in zip(images, axes, titles):
        axis.set_title(title)
        axis.imshow(image[:, :, 0])
        axis.axis('off')
    fig.show()


@tf.function
def train_step(generator: tf.keras.Model,
               generator_optimizer: tf.keras.optimizers.Optimizer,
               discriminator: tf.keras.Model,
               discriminator_optimizer: tf.keras.optimizers.Optimizer,
               loss_object: tf.keras.losses.Loss, inp_a: tf.Tensor,
               inp_b: tf.Tensor, target: tf.Tensor, epoch: int,
               summary_writer: tf.summary.SummaryWriter) -> None:
    """Trains the models for one (1) epoch.

  See https://www.tensorflow.org/tutorials/generative/pix2pix#training.

  Args:
    generator: A generator model,
    generator_optimizer: and an optimizer for the generator.
    discriminator: A discriminator model,
    discriminator_optimizer: and an optimizer for the generator.
    loss_object: A reusable BinaryCrossentropy object.
    inp_a: A full-width image of the left-most component.
    inp_b: A full-width image of the right-most component.
    target: The human-authored image of the a+b character.
    epoch: The index of the epoch we're in.
    summary_writer: A SummaryWriter object for writing.... summaries.
  """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        inp_x = tf.concat([inp_a, inp_b], 3)
        gen_output = generator(inp_x, training=True)

        disc_real_output = discriminator([inp_x, target], training=True)
        disc_generated_output = discriminator([inp_x, gen_output],
                                              training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            loss_object, disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(loss_object, disc_real_output,
                                       disc_generated_output)

    # TODO(ambuc): Should this simply be gen_l1_loss?
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(generator: tf.keras.Model,
        generator_optimizer: tf.keras.optimizers.Optimizer,
        discriminator: tf.keras.Model,
        discriminator_optimizer: tf.keras.optimizers.Optimizer,
        loss_object: tf.keras.losses.Loss, train_ds: tf.data.Dataset,
        epochs: int, test_ds: tf.data.Dataset, checkpoint: tf.train.Checkpoint,
        checkpoint_prefix: Text,
        summary_writer: tf.summary.SummaryWriter) -> None:
    """Runs for |epochs| and trains the models.

  Args:
    generator: A generator model,
    generator_optimizer: and an optimizer for the generator.
    discriminator: A discriminator model,
    discriminator_optimizer: and an optimizer for the generator.
    loss_object: A reusable BinaryCrossentropy object.
    train_ds:
    epochs: The number of epochs to train for.
    test_ds:
    checkpoint:
    checkpoint_prefix:
    summary_writer: A SummaryWriter object for writing.... summaries.
  """
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        for a, b, ab in test_ds.take(1):
            generate_images(generator, a, b, ab)
        print('Epoch: ', epoch)

        for n, (inp_a, inp_b, target) in train_ds.enumerate():
            print('.', end='')
            if (n + 1) % 100 == 0:
                print()
            train_step(generator, generator_optimizer, discriminator,
                       discriminator_optimizer, loss_object, inp_a, inp_b,
                       target, epoch, summary_writer)
        print()

        checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(
            epoch + 1,
            time.time() - start))
    checkpoint.save(file_prefix=checkpoint_prefix)
