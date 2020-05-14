from pythainlp.tokenize import word_tokenize
from pythainlp.util import thai_digit_to_arabic_digit
from pythainlp.corpus.common import thai_stopwords

import numpy as np
import re
import copy
import sys

sys.path.append('../config/hparams.py')


#from pythainlp.ulmfit import ThaiTokenizer


### PREPROCESS EACH STEP ###

# ? Not sure that Should this function use?
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


def remove_stopwords(tokenized_ls):
    removed_stopwords = []
    for text in tokenized_ls:
        tmp = []
        for word in text:
            if word not in thai_stopwords():
                tmp.append(word)
        removed_stopwords.append(tmp)
    return removed_stopwords


def use_first_n_words(tokenized_ls, n):
    return tokenized_ls[:n]
##############################

#! all steps preprocess


def preprocess_text(text, removed_stopwords=False):
    '''
    Preprocess all steps above 
    Input: news content (String)
    Output: ( 
            preprocessed tokenized news content (list), 
            preprocessed tokenized news content removed stopwords (list) 
            )
    '''
    hparams = hparams.create_hparams()

    text = remove_date(text)
    text = thai_digit_to_arabic_digit(text)
    text = basic_cleaner(text)

    tokenized_text = word_tokenize(text, engine='deepcut')
    tokenized_text = use_first_n_words(tokenized_text, n=hparams.maxlen)
    if remove_stopwords:
        removed_stopwords_text = remove_stopwords(tokenized_text)
        return remove_stopwords
    return tokenized_text

##### remove the first headline of this news #####
# # remove redundant sentence
# ls = []
# for i in range(len(df)):
#   text = df.iloc[i]['content'].replace(df.iloc[i]['headline'].strip(), "")
#   ls.append(text)
# df['content'] = ls
# df['content'] = df['content'].apply(strip_content)
##################################################
