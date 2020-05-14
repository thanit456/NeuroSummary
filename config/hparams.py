import tensorflow as tf
from tensorflow.keras.models import load_model


def create_hparams():
    hparams = tf.contrib.training.HParams(
        maxlen=50,
        maxlen_output=20
    )
    return hparams
