from services.inferencer import Inferencer
from config.hparams import create_hparams

from preprocessed.preprocessed_thaigov import preprocess_text, convert
from decode.word_decode import inference

hparams = create_hparams()


class TfIdfInferencer(Inferencer):

    def __init__(self):
        super().__init__()

    def infer(self, content):
        preprocessed_text = preprocess_text(
            content, hparams['removed_stopwords'])
        index_seq = convert(preprocessed_text)

        tf = {}
        idf = {}


        return inferred_headline
