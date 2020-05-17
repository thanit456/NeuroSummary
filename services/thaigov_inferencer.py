from services.inferencer import Inferencer
from config.hparams import create_hparams
from keras import models
from preprocessed.preprocessed_thaigov import preprocess_thaigov_text
from preprocessed.utils import word_to_index
from decode.word_decode import inference

hparams = create_hparams()

MODEL_WO_STOPWORDS_INFENC_PATH = './model/model_inf_units_64_batch_256_lr_0.01_drop_0.0/model_infenc_units_64_batch_256_lr_0.01_drop_0.0'
MODEL_WO_STOPWORDS_INFDEC_PATH = './model/model_inf_units_64_batch_256_lr_0.01_drop_0.0/model_infdec_units_64_batch_256_lr_0.01_drop_0.0'
MODEL_W_STOPWORDS_INFENC_PATH = './model/model_stop_word_inf_units_64_batch_256_lr_0.01_drop_0.0_val_acc_save_best/model_stop_word_infenc_units_64_batch_256_lr_0.01_drop_0.0_val_acc_save_best'
MODEL_W_STOPWORDS_INFDEC_PATH = './model/model_stop_word_inf_units_64_batch_256_lr_0.01_drop_0.0_val_acc_save_best/model_stop_word_infdec_units_64_batch_256_lr_0.01_drop_0.0_val_acc_save_best'

if hparams['removed_stopwords']:
    infenc_path = MODEL_W_STOPWORDS_INFENC_PATH
    infdec_path = MODEL_W_STOPWORDS_INFDEC_PATH
else:
    infenc_path = MODEL_WO_STOPWORDS_INFENC_PATH
    infdec_path = MODEL_WO_STOPWORDS_INFDEC_PATH


class ThaigovInferencer(Inferencer):

    def __init__(self):
        super().__init__()

        self.infenc = models.load_model(infenc_path)
        self.infdec = models.load_model(infdec_path)

    def infer(self, content):
        preprocessed_text = preprocess_thaigov_text(
            content, hparams['removed_stopwords'])
        index_seq = word_to_index(preprocessed_text)
        inferred_headline = inference(
            [index_seq], self.infenc, self.infdec, hparams['removed_stopwords'])
        return inferred_headline
