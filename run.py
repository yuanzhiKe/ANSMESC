import pickle
import os
import keras
import sklearn
import numpy as np
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
from sklearn.model_selection import KFold
from cnn_with_idc import build_model

MAX_SENTENCE_LENGTH = 500
MAX_WORD_LENGTH = 4
COMP_WIDTH = 3
CHAR_EMB_DIM = 15
BATCH_SIZE = 64
VERBOSE = 1
EPOCHS = 40
POS = False
CNN_ENCODER = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    kf = KFold(n_splits=5)
    X = x_sip_train
    y = y_train
    for n, (train_index, test_index) in enumerate(kf.split(X)):
        print(f'****************Iteration {n}********************')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model_name = f"Radical-CNN-RNN_HARC_POS_ADD_RIDGE_BN_RMSPROP0_CROSS{n}"
        K.clear_session()
        # using the vocab size of the example data
        model = build_model(radical_vocab_size=2487, char_vocab_size=21294,
                            max_sentence_length=MAX_SENTENCE_LENGTH, max_word_length=MAX_WORD_LENGTH,
                            comp_width=COMP_WIDTH, char_emb_dim=CHAR_EMB_DIM, classes=2,
                            char_shape=True, word=False, char=False,
                            cnn_encoder=True, highway="relu", nohighway="linear",
                            attention=True, shape_filter=True, char_filter=True, position=True)
        train_model(model, X_train, y_train, X_test, y_test, model_name, path="./")
        test_model(model, X_test, y_test)
