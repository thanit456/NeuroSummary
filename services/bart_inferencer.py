from services.inferencer import Inferencer
from config.hparams import create_hparams
from preprocessed import preprocess_any_text

from transformers import pipeline
from googletrans import Translator

hparams = create_hparams()


class BartInferencer(Inferencer):

    def __init__(self):
        super().__init__()

    def infer(self, content):

        if hparams['removed_stopwords']:
            preprocessed_text = preprocess_any_text(
                content, hparams['removed_stopwords'])
            th_input = ''.join(preprocessed_text)
        else:
            th_input = content

        translator = Translator()
        en_input = translator.translate(th_input, dest='en').text

        summarizer = pipeline('summarization')
        en_summary = summarizer(en_input, min_length=10,
                                max_length=hparams['maxlen_output'])

        #! When translate to thai, it cannot make sure that max_length is still 20.
        th_summary = translator.translate(en_summary, dest='th').text

        return th_summary
