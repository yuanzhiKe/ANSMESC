import pickle
import os
import keras
import sklearn
import numpy as np
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
from sklearn.model_selection import KFold
from multi_joint import build_model

MAX_SENTENCE_LENGTH = 500
MAX_WORD_LENGTH = 4
COMP_WIDTH = 3
CHAR_EMB_DIM = 15
BATCH_SIZE = 64
WORD_DIM = 600
VERBOSE = 1
EPOCHS = 40
POS = False
CNN_ENCODER = False
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train_model(model, x_train, y_train, x_val, y_val, model_name, early_stop=False, path="", epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE):
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=VERBOSE, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    if early_stop:
        stopper = EarlyStopping(monitor='val_loss', patience=10)
    checkpoint_loss = ModelCheckpoint(filepath=path + "checkpoints/" + model_name + "_loss.hdf5",
                                      monitor="val_loss",
                                      verbose=verbose, save_best_only=False, mode="min")
    opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    print("compling...")
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['categorical_crossentropy', "acc"], )
    print("fitting...")
    if early_stop:
        result = model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=verbose,
                           epochs=epochs, batch_size=batch_size, callbacks=[reducelr, stopper, checkpoint_loss])
    else:
        result = model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=verbose,
                           epochs=epochs, batch_size=batch_size, callbacks=[reducelr, checkpoint_loss])
    return result

def max_from_category(y):
    y_true = np.zeros(y.shape[0], dtype=np.int)
    for i, v in enumerate(y):
        y_true[i] = np.argmax(v)
    return y_true

def test_model(model, x_test, y_test):
    print("testing...")
    y_true = max_from_category(y_test)
    y_pred = model.predict(x_test, verbose=0)
    y_pred = max_from_category(y_pred)
    print(sklearn.metrics.classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    with open('RakutenSubset4exp.pickle', 'rb') as f:
        x_sip_train, x_sip_validation, x_sip_test_normal, x_sip_test_unk_w, x_sip_test_unk_c, y_train, y_validation, y_test_normal, y_test_unk_w, y_test_unk_c = pickle.load(f)

    with open("../CMWE/unk_exp/rakuten_processed_review_split_ongly15.pickle", "rb") as f:
        full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin, \
        preprocessed_char_number, word_vocab, char_vocab, \
        x_s_train, x_c_train, x_w_train, y_train, \
        x_s_validation, x_c_validation, x_w_validation, y_validation, \
        x_s_test_normal, x_c_test_normal, x_w_test_normal, y_test_normal, \
        x_s_test_unk_w, x_c_test_unk_w, x_w_test_unk_w, y_test_unk_w, \
        x_s_test_unk_c, x_c_test_unk_c, x_w_test_unk_c, y_test_unk_c = pickle.load(f)
    word_vocab_size = len(word_vocab)
    char_vocab_size = len(char_vocab)
 
    kf = KFold(n_splits=5)
    X = x_sip_train
    Xw = x_w_train
    Xc = x_c_train
    y = y_train
    model_name = f"Radical-ONLY-RMSPROP0"
    K.clear_session()
    # using the vocab size of the example data
    model = build_model(radical_vocab_size=real_vocab_number, char_vocab_size=char_vocab_size, word_vocab_size = word_vocab_size, max_sentence_length=MAX_SENTENCE_LENGTH, max_word_length=MAX_WORD_LENGTH, classes=2, word_dim=WORD_DIM, comp_width=COMP_WIDTH, char=True, word=True)
    train_model(model, X, y_train, x_sip_validation, y_validation, model_name, path="./")
    # model.load_weights("checkpoints/" + model_name + "_loss.hdf5")
    test_model(model, x_sip_test_normal, y_test_normal)
    test_model(model, x_sip_test_unk_w, y_test_unk_w)
    test_model(model, x_sip_test_unk_c, y_test_unk_c)
    
