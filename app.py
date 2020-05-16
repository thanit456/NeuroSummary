from flask import abort, request, Flask, jsonify
from keras import models
import argparse

from preprocessed.preprocessed_thaigov import preprocess_text, convert
from decode.word_decode import inference
from config.hparams import create_hparams

import numpy as np

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
    preprocessed_text = preprocess_text(request.json['content'])
    return jsonify({'preprocessed_text': preprocessed_text}), 201


# fully inference headline generation with word-based
@app.route('/inference', methods=['POST'])
def infer():
    if not request.json or not 'content' in request.json:
        abort(400)
    preprocessed_text = preprocess_text(request.json['content'])
    index_seq = convert(preprocessed_text)

    infenc = models.load_model(args.infenc_path)
    infdec = models.load_model(args.infdec_path)

    inferred_headline = inference([index_seq], infenc, infdec, hparams['removed_stopwords'])
    return jsonify({'inferred_headline': inferred_headline}), 201


if __name__ == '__main__':
    app.run(debug=True)
