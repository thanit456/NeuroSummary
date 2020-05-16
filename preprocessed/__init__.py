from pythainlp.tokenize import word_tokenize

from preprocessed.preprocessed_thaigov import preprocess_thaigov_text
from preprocessed.utils import remove_stopwords


def preprocess_text(text, dataset, removed_stopwords=False):
    if dataset == "thaigov":
        preprocessed_text = preprocess_thaigov_text(text, removed_stopwords)
    else:
        preprocessed_text = preprocess_any_text(text, removed_stopwords)
    return preprocessed_text


######### Other preprocess ######

def preprocess_any_text(text, removed_stopwords=False):
    ''' Only tokenize text with deepcut
    '''
    tokenized_text = word_tokenize(text, engine='deepcut')
    if not removed_stopwords:
        return tokenized_text
    return remove_stopwords(tokenized_text)

#################################
