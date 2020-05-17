from services.inferencer import Inferencer
from config.hparams import create_hparams
from preprocessed import preprocess_any_text

from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer
from googletrans import Translator

hparams = create_hparams()


class GPT2Inferencer(Inferencer):

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

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        ids = tokenizer.encode(en_input, return_tensors='tf')
        # ? maybe use ByteLebelBPETOkenizer for fast tokenization
        # ! still not implemented
        # tokenizer.save_pretrained('../decode/gpt2')
        #fast_tokenizer = ByteLevelBPETokenizer('../decode/gpt2/vocab.json', '../decode/gpt2/merge.txt', lowercase=True)
        #ids = fast_tokenizer.encode(en_input)

        model = TFGPT2LMHeadModel.from_pretrained('gpt2-large')

        sample_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=20,
            top_k=55,
            top_p=0.95,
            num_return_sequences=5
        )

        inference_outputs = []
        for sample_output in sample_outputs:
            out = tokenizer.decode(sample_output, skip_special_tokens=True)
            new_out = translator.translate(out, dest='th').text
            inference_outputs.append(new_out)

        return inference_outputs
