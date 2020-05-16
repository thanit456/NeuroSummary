import tensorflow as tf
from tensorflow.keras.models import load_model


def create_hparams():
    hparams = {
        "maxlen": 50,
        "maxlen_output": 20,
        "removed_stopwords": False,
    }
    return hparams
