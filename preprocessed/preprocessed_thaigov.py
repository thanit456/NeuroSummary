import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K

import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
import copy

#from pythainlp.ulmfit import ThaiTokenizer
from pythainlp.corpus.common import thai_stopwords
from pythainlp.util import thai_digit_to_arabic_digit
from pythainlp.tokenize import word_tokenize

df = pd.read_csv(
    '/content/drive/Shared drives/NeuroSummary/data/1_24_thaigov.csv')

df = df.rename(columns={'content': 'raw_content'})
df.head()

### PREPROCESS EACH STEP ###


def extract_most_specific_headline(text):
    text = text[text.find('-') + 1:]
    return text[text.find('-') + 1:]


def remove_date(text):
    for _ in range(4):
        text = text[text.find(' ') + 1:]
    return text


def basic_cleaner(text):
    # type(result)
    special_char = '!|"|#|%|&|\'|,|-|/|:|;|<|=|>|@|_|`|~|\.|\$'.split("|")
    nstr = text
    for special in special_char:
        nstr = re.sub(special+"+", special.replace('\\', ''), nstr)
    nstr = re.sub("|^||'", "", nstr).replace("|", "").replace(
        "*", "").replace("  ", " ").replace(u'\xa0', u' ').replace(u'\u200b', u' ')
    return nstr


def strip_each_word(tokens):
    ls = []
    for token in tokens:
        ls.append(token.strip())
    return ls

# remove stopwords


def remove_stopwords(tokenized_ls):
    removed_stopwords = []
    for text in tqdm_notebook(tokenized_ls):
        tmp = []
        for word in text:
            if word not in thai_stopwords():
                tmp.append(word)
        removed_stopwords.append(tmp)
    return removed_stopwords
##############################

#! all steps preprocess


def preprocess_text(text):
    '''
    Preprocess all steps above 
    Input: news content (String)
    Output: ( 
            preprocessed tokenized news content (list), 
            preprocessed tokenized news content removed stopwords (list) 
            )
    '''
    text = remove_date(text)
    text = thai_digit_to_arabic_digit(text)
    text = basic_cleaner(text)

    tokenized_text = word_tokenize(text, engine='deepcut')

    removed_stopwords_text = remove_stopwords(text)

    return tokenized_text, removed_stopwords_text

##### remove the first headline of this news #####
# # remove redundant sentence
# ls = []
# for i in range(len(df)):
#   text = df.iloc[i]['content'].replace(df.iloc[i]['headline'].strip(), "")
#   ls.append(text)
# df['content'] = ls
# df['content'] = df['content'].apply(strip_content)
##################################################
