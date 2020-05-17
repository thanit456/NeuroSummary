from flask import abort, request, Flask, jsonify

from preprocessed import preprocess_text
from decode.word_decode import inference
from config.hparams import create_hparams

from services.thaigov_inferencer import ThaigovInferencer
from services.bart_inferencer import BartInferencer
from services.t5_inferencer import T5Inferencer
from services.gpt2_inferencer import GPT2Inferencer

import tensorflow as tf

inferencers = {
    'thaigov': ThaigovInferencer(),
    'bart': BartInferencer(),
    't5': T5Inferencer(),
    'gpt2': GPT2Inferencer
}

hparams = create_hparams()
graph = tf.get_default_graph()

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return 'It works!', 200

#! Just test not for fully used
@app.route('/preprocess_thaigov', methods=['POST'])
def preprocess():

    if not request.json or not 'content' in request.json:
        abort(400)

    content = request.json['content']

    preprocessed_text = preprocess_text(content, hparams['removed_stopwords'])

    return jsonify({'preprocessed_text': preprocessed_text}), 201


# fully inference headline generation with word-based
@app.route('/inference/<inf_name>', methods=['POST'])
def infer(inf_name='thaigov'):
    if not request.json or not 'content' in request.json:
        abort(400)

    if inf_name not in inferencers:
        abort(404)

    content = request.json['content']

    global graph
    with graph.as_default():
        inferred_headline = inferencers[inf_name].infer(content)

    return jsonify({'inferred_headline': inferred_headline}), 201


if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
