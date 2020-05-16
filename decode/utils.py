import pickle


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
