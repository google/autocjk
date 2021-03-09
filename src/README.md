# Modiying the generator

## How to generate training data:

Our dataset is a series of images which look like: "吉乚㐖".

Example:

![U+3416](../docs/images/0000022.png)

1. Download some fonts. For example, you might choose to train on the set of
   Noto CJK fonts.

1. Once you have a list of local font files, run:

   ```bash
   # Make a directory. The training images will get written her.
   mkdir /tmp/corpus
   ```

   ```bash
   # Run :make_dataset. I use fish-style subcommands: (); you might need
   # bash-style: $().
   bazel run //src:make_dataset -- \
     --dst=/tmp/corpus \
     --fonts=(string join ',' (ls ~/Downloads/Noto{Sans,Serif}CJKsc-*))
   ```

## How to retrain the generator:

Ideally you have access to a notebook-style environment running remotely on a
machine suitable for ML training. I use Colab-style magics here.

1. You'll need to copy the training data generated in the last step to a
   local directory.

1. In the notebook, set up all the Tensorflow machinery.

   ```python
   import tensorflow as tf
   import datetime
   from autocjk.src import model as model_lib

   train_dataset, test_dataset = model_lib.make_datasets('/tmp/corpus/*.png')

   generator = model_lib.make_generator()
   discriminator = model_lib.make_discriminator()

   loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

   generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
   discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

   checkpoint_dir = './training_checkpoints'
   checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
   checkpoint = tf.train.Checkpoint(
       generator_optimizer=generator_optimizer,
       discriminator_optimizer=discriminator_optimizer,
       generator=generator,
       discriminator=discriminator)

   summary_writer = tf.summary.create_file_writer(
       'logs/' + 'fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
   ```

- Kick off the training. You can run this for one epoch just to see if it
  works, or for more for a more accurate model.

  ```python
  _EPOCHS = 50
  model_lib.fit(generator, generator_optimizer, discriminator,
            discriminator_optimizer, loss_object, train_dataset, _EPOCHS,
            test_dataset, checkpoint, checkpoint_prefix, summary_writer)
  ```

- You can pause training at any time to evaluate the model manually.

  ```python
  for a, b, ab in test_dataset.take(1):
      model_lib.generate_images(generator, a, b, ab)
  ```

  You can pull data from `test_dataset` for a guarantee that the model didn't
  train on them. For more assurance, you can pass arguments like so:
  `model_lib.generate_images(generator, b, a, ab)`. This will flip the LHS and
  RHS and produce a character which (probably) doesn't exist in any dataset.
  Watch out: if the LHS character is half-width (like '亻'), your output will
  probably be nonsense.

- When you're ready, save the model locally:

  ```python
  generator.save('/tmp/generator.h5')

  ```

Filecopy the generator to your local machine and use it with `bazel run //src:main`, as described in `../README.md`.

## Validating results with out-of-band predictions.

Use `//src:evaluate_glyphwiki` like so:

```bash
mkdir /tmp/output
bazel run //src:evaluate_glyphwiki -- \
    --alsologtostderr \
    --font_path=<path_to_reference_font> \
    --limit=10 \
    --out_dir=/tmp/output
```

You can view the prediced image(s) in the output directory, or raise the limit
to evaluate more characters at once.
