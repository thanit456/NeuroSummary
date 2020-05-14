import tensorflow as tf


def create_hparams():
    hparams = tf.contrib.training.HParams(
        maxlen=50,
        maxlen_output=20,
    )
    return hparams
