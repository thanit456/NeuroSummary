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

from pythainlp.ulmfit import ThaiTokenizer
from pythainlp.corpus.common import thai_stopwords
from pythainlp.util import thai_digit_to_arabic_digit


df = pd.read_csv('/content/drive/Shared drives/NeuroSummary/data/1_24_thaigov.csv')

df = df.rename(columns={'content': 'raw_content'})
df.head()

### PREPROCESS IN DATAFRAME ###
def extract_most_specific_headline(text):
  text = text[text.find('-') + 1:]
  return text[text.find('-') + 1:]

def remove_date(text):
  for _ in range(4):
    text = text[text.find(' ') + 1:]
  return text

def extract_date(text):
  cumulative_idx = 0
  tmp_text = copy.copy(text)
  for _ in range(4):
    idx = tmp_text.find(' ')
    tmp_text = tmp_text[idx + 1:]
    cumulative_idx += (idx + 1)
  return text[: cumulative_idx - 1]

def basic_cleaner(text) :
  special_char = '!|"|#|%|&|\'|,|-|/|:|;|<|=|>|@|_|`|~|\.|\$'.split("|")# type(result)
  nstr = text
  for special in special_char :
    nstr = re.sub(special+"+",special.replace('\\',''),nstr)
  nstr = re.sub("|^||'","",nstr).replace("|","").replace("*","").replace("  "," ").replace(u'\xa0', u' ').replace(u'\u200b', u' ')
  return nstr

def strip_content(text):
  return text.strip()

df['specific_headline'] = df['headline'].apply(extract_most_specific_headline)
df['date'] = df['raw_content'].apply(extract_date)
df['content'] = df['raw_content'].apply(remove_date)
df['content'] = df['content'].apply(basic_cleaner)
df['content'] = df['content'].apply(thai_digit_to_arabic_digit)
df['specific_headline'] = df['specific_headline'].apply(basic_cleaner)
df['specific_headline'] = df['specific_headline'].apply(thai_digit_to_arabic_digit)
df = df.drop('headline', axis=1)
df = df.rename(columns={ 'specific_headline': 'headline'})

# remove redundant sentence
ls = []
for i in range(len(df)):
  text = df.iloc[i]['content'].replace(df.iloc[i]['headline'].strip(), "")
  ls.append(text)
df['content'] = ls
df['content'] = df['content'].apply(strip_content)
##############

from pythainlp.tokenize import word_tokenize

# ulmfitTokenizer = ThaiTokenizer()

def preprocessed(df, column='content'):
  tokenized_ls = []
  for text in tqdm_notebook(df[column]):
    tokenized_text = word_tokenize(text, engine='deepcut')
    tokenized_ls.append(tokenized_text)
  return tokenized_ls

tokenized_content_ls = preprocessed(df)
tokenized_headline_ls = preprocessed(df, column='headline')

df['tokenized_deepcut_content'] = tokenized_content_ls
df['tokenized_deepcut_headline'] = tokenized_headline_ls

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

removed_stopwords_contents = remove_stopwords(tokenized_content_ls)
removed_stopwords_headlines = remove_stopwords(tokenized_headline_ls)
df['removed_stopwords_content'] = removed_stopwords_contents
df['removed_stopwords_headline'] = removed_stopwords_headlines

# seperate class_name from file_name
ls = []
for file_name in df['file_name']:
  class_name = file_name[file_name.find('/') + 1: file_name.find('_')]
  ls.append(class_name)
df['class_name'] = ls