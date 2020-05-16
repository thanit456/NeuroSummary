import re
import numpy as np

from pythainlp.util import thai_digit_to_arabic_digit
from pythainlp.corpus.common import thai_stopwords

from decode.utils import create_dictionary


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


def word_to_index(input_data):
    dict_t, _, _ = create_dictionary()

    data = []
    for word in input_data:
        if word in dict_t:
            data.append(dict_t[word])
        else:
            data.append(dict_t["UNK"])
    return np.array(data)
