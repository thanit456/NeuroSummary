from preprocessed.preprocessed_thaigov import preprocess_thaigov_text
from preprocessed.preprocessed_cmu import preprocess_cmu_text


def preprocess_text(text, dataset):
    if dataset == "thaigov":
        preprocessed_text = preprocess_thaigov_text(text)
    elif dataset == "preprocess_cmu_text":
        preprocessed_text = preprocess_cmu_text(text)
    return preprocessed_text
