from flask import abort, request, Flask, jsonify
from keras import models
import argparse
import numpy as np

from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, AutoModelWithLMHead, AutoTokenizer, pipeline
from tokenizers import ByteLevelBPETokenizer
from googletrans import Translator

from preprocessed.utils import word_to_index
from preprocessed import preprocess_text
from decode.word_decode import inference
from config.hparams import create_hparams

hparams = create_hparams()

MODEL_WO_STOPWORDS_INFENC_PATH = './model/model_inf_units_64_batch_256_lr_0.01_drop_0.0/model_infenc_units_64_batch_256_lr_0.01_drop_0.0'
MODEL_WO_STOPWORDS_INFDEC_PATH = './model/model_inf_units_64_batch_256_lr_0.01_drop_0.0/model_infdec_units_64_batch_256_lr_0.01_drop_0.0'
MODEL_W_STOPWORDS_INFENC_PATH = './model/model_stop_word_inf_units_64_batch_256_lr_0.01_drop_0.0_val_acc_save_best/model_stop_word_infenc_units_64_batch_256_lr_0.01_drop_0.0_val_acc_save_best'
MODEL_W_STOPWORDS_INFDEC_PATH = './model/model_stop_word_inf_units_64_batch_256_lr_0.01_drop_0.0_val_acc_save_best/model_stop_word_infdec_units_64_batch_256_lr_0.01_drop_0.0_val_acc_save_best'

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--infenc_path', default='')
parser.add_argument('-d', '--infdec_path', default='')
# parser.add_argument('-r', '--removed_stopwords', default=False)
args = parser.parse_args()

if hparams['removed_stopwords']:
    args.infenc_path = MODEL_W_STOPWORDS_INFENC_PATH
    args.infdec_path = MODEL_W_STOPWORDS_INFDEC_PATH
else:
    args.infenc_path = MODEL_WO_STOPWORDS_INFENC_PATH
    args.infdec_path = MODEL_WO_STOPWORDS_INFDEC_PATH

app = Flask(__name__)


#! Just test not for fully used
@app.route('/preprocess', methods=['POST'])
def preprocess():

    if not request.json or not 'content' in request.json:
        abort(400)

    content = request.json['content']
    dataset = request.json['dataset']

    preprocessed_text = preprocess_text(content, dataset)

    return jsonify({'preprocessed_text': preprocessed_text}), 201


# fully inference headline generation with word-based
@app.route('/inference', methods=['POST'])
def infer():

    if not request.json or not 'content' in request.json:
        abort(400)

    content = request.json['content']
    dataset = request.json['dataset']

    preprocessed_text = preprocess_text(content, dataset)

    index_seq = word_to_index(preprocessed_text)

    infenc = models.load_model(args.infenc_path)
    infdec = models.load_model(args.infdec_path)

    inferred_headline = inference(
        [index_seq], infenc, infdec, hparams['removed_stopwords'])

    return jsonify({'inferred_headline': inferred_headline}), 201


@app.route('/gpt2_infer', methods=['POST'])
def gpt2_infer():

    if not request.json or not 'content' in request.json:
        abort(400)

    content = request.json['content']
    dataset = request.json['dataset']

    if dataset == 'thaigov':
        preprocessed_text = preprocess_text(content, dataset)
        th_input = ''.join(preprocessed_text)
    else:
        th_input = content
    translator = Translator()
    en_input = translator.translate(th_input, dest='en').text

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    ids = tokenizer.encode(en_input, return_tensors='tf')
    # ? maybe use ByteLebelBPETOkenizer for fast tokenization
    # ! still not implemented
    # tokenizer.save_pretrained('.')
    #fast_tokenizer = ByteLevelBPETokenizer('./vocab.json', './merge.txt', lowercase=True)
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

    return jsonify({'inferred_headline': inference_outputs}), 201


@app.route('/bart_infer', methods=['POST'])
def bart_infer():
    if not request.json or not 'content' in request.json:
        abort(400)

    content = request.json['content']
    dataset = request.json['dataset']

    if dataset == 'thaigov':
        preprocessed_text = preprocess_text(content, dataset)
        th_input = ''.join(preprocessed_text)
    else:
        th_input = content

    translator = Translator()
    en_input = translator.translate(th_input, dest='en').text

    summarizer = pipeline('summarization')
    en_summary = summarizer(en_input, min_length=10, max_length=20)

    #! When translate to thai, it cannot make sure that max_length is still 20.
    th_summary = translator.translate(en_summary, dest='th').text
    return jsonify({'inferred_headline': th_summary}), 201


@app.route('/t5_infer', methods=['POST'])
def t5_infer():
    if not request.json or not 'content' in request.json:
        abort(400)

    content = request.json['content']
    dataset = request.json['dataset']

    if dataset == 'thaigov':
        preprocessed_text = preprocess_text(content, dataset)
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
    outputs = model.generate(ids, max_length=20, min_length=10,
                             length_penalty=2.0, num_beams=4, early_stopping=True)
    th_outputs = [e for e in outputs]
    return jsonify({'inferred_headline': th_outputs}), 201


if __name__ == '__main__':
    app.run(debug=True)
