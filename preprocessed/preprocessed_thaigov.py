from pythainlp.tokenize import word_tokenize


from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import re
import copy
import sys

from config.hparams import create_hparams
from decode.utils import create_dictionary
from preprocessed.utils import remove_date, thai_digit_to_arabic_digit, basic_cleaner, use_first_n_words, remove_stopwords, word_to_index


def preprocess_thaigov_text(text, removed_stopwords=False):
    '''
    Preprocess all steps above 
    Input: news content (String)
    Output: ( 
            preprocessed tokenized news content (list), 
            preprocessed tokenized news content removed stopwords (list) 
            )
    '''
    hparams = create_hparams()

    text = remove_date(text)
    text = thai_digit_to_arabic_digit(text)
    text = basic_cleaner(text)

    tokenized_text = word_tokenize(text, engine='deepcut')
    tokenized_text = use_first_n_words(tokenized_text, n=hparams['maxlen'])
    if removed_stopwords:
        removed_stopwords_text = remove_stopwords(tokenized_text)
        return removed_stopwords_text
    return tokenized_text
