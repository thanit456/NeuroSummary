import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys
import os

#! may be use absolute path
from config.hparams import create_hparams


def create_dictionary(removed_stopwords=False):
    if not removed_stopwords:
        with open('./decode/dictionary/dict.pkl', 'rb') as f:
            dict_t = pickle.load(f)
        with open('./decode/dictionary/rev_dict.pkl', 'rb') as f:
            rev_dict_t = pickle.load(f)
    else:
        with open('./decode/dictionary/stop_dict.pkl', 'rb') as f:
            dict_t = pickle.load(f)
        with open('./decode/dictionary/stop_rev_dict.pkl', 'rb') as f:
            rev_dict_t = pickle.load(f)
    return dict_t, rev_dict_t, len(dict_t)


def predict_sequence(infenc, infdec, source, n_steps, removed_stopwords=False):

    dict_t, rev_dict_t, vocab_size = create_dictionary()

    #! Be able to change if it doesn't work
    num_decoders = vocab_size

    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = [dict_t['<s>']]
    decoded_sentence = ''
    for t in range(n_steps):
        # predict next char
        output_tokens, h, c = infdec.predict([target_seq] + state)
        # store prediction
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = rev_dict_t[sampled_token_index]
        if(sampled_char == "</s>"):
            if(len(decoded_sentence) == 0):
                return "</s>"
            return decoded_sentence
        decoded_sentence += " " + sampled_char
        # update state &  target sequence
        target_seq = [sampled_token_index]
        state = [h, c]

    return decoded_sentence


def inference(dataset, inf_enc, inf_dec, removed_stopwords=False):

    dict_t, rev_dict_t, vocab_size = create_dictionary()

    #! Be able to change if it doesn't work
    hparams = create_hparams()
    steps = hparams['maxlen_output']
    maxlen = hparams['maxlen']

    pred = predict_sequence(inf_enc, inf_dec, pad_sequences(
        dataset, maxlen=maxlen, padding='post'), steps, removed_stopwords=removed_stopwords)

    return pred.strip()
