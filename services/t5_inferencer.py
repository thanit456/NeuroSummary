from services.inferencer import Inferencer
from config.hparams import create_hparams
from preprocessed import preprocess_any_text

from transformers import AutoModelWithLMHead, AutoTokenizer
from googletrans import Translator

hparams = create_hparams()


class T5Inferencer(Inferencer):

    def __init__(self):
        super().__init__()

    def infer(self, content):

        if hparams['removed_stopwords']:
            preprocessed_text = preprocess_any_text(
                content, hparams['removed_stopwords'])
            th_input = ''.join(preprocessed_text)
        else:
            th_input = content

        model = AutoModelWithLMHead.from_pretrained("t5-base")
        tokenizer = AutoTokenizer.from_pretrained("t5-base")

        translator = Translator()
        en_input = translator.translate(th_input, dest='en').text
        # ? max_length = 512 may be able to change
        ids = tokenizer.encode("summarize: "+th_input,
                               return_tensors='pt', max_length=512)
        # ! I'm not sure that whether it has to decode or not
        outputs = model.generate(ids, max_length=hparams['maxlen_output'], min_length=10,
                                 length_penalty=2.0, num_beams=4, early_stopping=True)
        th_outputs = [e for e in outputs]

        return th_outputs
