from pythainlp.tokenize import word_tokenize

from preprocessed.utils import remove_stopwords

######### Other preprocess ######


def preprocess_any_text(text, removed_stopwords=False):
    ''' Only tokenize text with deepcut
    '''
    tokenized_text = word_tokenize(text, engine='deepcut')
    if not removed_stopwords:
        return tokenized_text
    return remove_stopwords(tokenized_text)

#################################
