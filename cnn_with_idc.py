import pickle
import os
import keras
import tensorflow as tf
from keras.layers import Dropout, BatchNormalization, Activation
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.models import Model
from keras.layers import Embedding, Input, AveragePooling1D, MaxPooling1D, Conv1D, concatenate, TimeDistributed, \
    Bidirectional, LSTM, Dense, Flatten, GRU, Lambda, Concatenate
from keras.legacy.layers import Highway
from attention import AttentionWithContext
from tqdm import tqdm


def initial_emb(vocab_size, emb_dim):
    zero_pad = np.zeros((1, emb_dim))
    limit = np.sqrt(1/(vocab_size + emb_dim))
    embedding_matrix =  np.random.uniform(-limit,limit,(vocab_size-1, emb_dim))
    w = np.vstack((zero_pad, embedding_matrix))
    return w

def build_word_feature_shape(vocab_size=5, char_emb_dim=15, comp_width=3, max_word_length=4,
                             mode="padding", cnn_encoder=True,
                             highway="linear", nohighway=None, shape_filter=True, char_filter=True, position=True):
    # build the feature computed by cnn for each word in the sentence. used to input to the next rnn.
    # expected input: every #comp_width int express a character.
    # mode:
    # "average": average pool the every #comp_width input embedding, output average of the indexed embeddings of a character
    # "padding": convoluate every #comp_width embedding

    # real vocab_size for ucs is 2481, including paddingblank, unkown, puncutations, kanas
    print('build word feature shape')
    assert shape_filter or char_filter
    init_width = 0.5 / char_emb_dim
    init_weight = np.random.uniform(low=-init_width, high=init_width, size=(vocab_size, char_emb_dim))
    init_weight[0] = 0  # maybe the padding should not be zero
    # print(init_weight)
    # first layer embeds
    #  every components
    num_inputs = 3
    if position:
        word_input = Input(shape=(num_inputs, comp_width * max_word_length))
        shape = Lambda(lambda x: x[:, 0, :])(word_input)  # use lambda to slice. not using lambda lead to errors
        idc = Lambda(lambda x: x[:, 1, :])(word_input)
        pos = Lambda(lambda x: x[:, 2, :])(word_input)
        shape_embedding = \
            Embedding(input_dim=vocab_size, output_dim=char_emb_dim, weights=[init_weight], trainable=True)(shape)
        init_weight_idc = np.random.uniform(low=-init_width, high=init_width, size=(13 + 1, char_emb_dim))
        init_weight_idc[0] = 0
        init_weight_pos = np.random.uniform(low=-init_width, high=init_width, size=(comp_width + 2, char_emb_dim))
        init_weight_pos[0] = 0
        idc_embedding = \
            Embedding(input_dim=13 + 1, output_dim=char_emb_dim, weights=[init_weight_idc], trainable=True)(
                idc)  # totally 12 + simple character idc
        pos_embedding = \
            Embedding(input_dim=comp_width + 2, output_dim=char_emb_dim, weights=[init_weight_pos], trainable=True)(pos)
#         char_embedding = Concatenate(axis=2)([shape_embedding, idc_embedding, pos_embedding])
        print('Add token embeddings and pos embeddings.')
        char_embedding = keras.layers.Add()([shape_embedding, idc_embedding, pos_embedding])
    else:
        word_input = Input(shape=(num_inputs, comp_width * max_word_length,))
        shape = Lambda(lambda x: x[:, 0, :])(word_input)  # use lambda to slice. not using lambda lead to errors
        shape_embedding = \
            Embedding(input_dim=vocab_size, output_dim=char_emb_dim, weights=[init_weight], trainable=True)(shape)
        char_embedding = shape_embedding
    # print("char_embedding:", char_embedding._keras_shape)
    if cnn_encoder:
        if mode == "padding":
            # print(char_embedding._keras_shape)
            # print(comp_width)
            if shape_filter and char_filter:
                filter_sizes = [50, 100, 150]
            else:
                filter_sizes = [100, 200, 300]
            if shape_filter:
                feature_s1 = Conv1D(filters=filter_sizes[0], kernel_size=1, activation='relu')(
                    char_embedding)
                feature_s1 = BatchNormalization()(feature_s1)
                feature_s1 = MaxPooling1D(pool_size=max_word_length * comp_width)(feature_s1)
                feature_s2 = Conv1D(filters=filter_sizes[1], kernel_size=2, activation='relu')(
                    char_embedding)
                feature_s2 = BatchNormalization()(feature_s2)
                feature_s2 = MaxPooling1D(pool_size=max_word_length * comp_width - 1)(feature_s2)
                feature_s3 = Conv1D(filters=filter_sizes[2], kernel_size=3, activation='relu')(
                    char_embedding)
                feature_s3 = BatchNormalization()(feature_s3)
                feature_s3 = MaxPooling1D(pool_size=max_word_length * comp_width - 2)(feature_s3)
            if char_filter:
                feature1 = Conv1D(filters=filter_sizes[0], kernel_size=1 * comp_width, strides=comp_width,
                                  activation='relu')(
                    char_embedding)
                feature1 = BatchNormalization()(feature1)
                feature1 = MaxPooling1D(pool_size=max_word_length - 1 + 1)(feature1)
                feature2 = Conv1D(filters=filter_sizes[1], kernel_size=2 * comp_width, strides=comp_width,
                                  activation='relu')(
                    char_embedding)
                feature2 = BatchNormalization()(feature2)
                feature2 = MaxPooling1D(pool_size=max_word_length - 2 + 1)(feature2)
                feature3 = Conv1D(filters=filter_sizes[2], kernel_size=3 * comp_width, strides=comp_width,
                                  activation='relu')(
                    char_embedding)
                feature3 = BatchNormalization()(feature3)
                feature3 = MaxPooling1D(pool_size=max_word_length - 3 + 1)(feature3)
            if shape_filter and char_filter:
                feature = Concatenate()([feature_s1, feature_s2, feature_s3, feature1, feature2, feature3])
            elif shape_filter and not char_filter:
                feature = Concatenate()([feature_s1, feature_s2, feature_s3])
            elif char_filter and not shape_filter:
                feature = Concatenate()([feature1, feature2, feature3])
            else:
                feature = None
        feature = Flatten()(feature)
        # print(feature._keras_shape)
#         if highway:
#             if isinstance(highway, str):
#                 feature = Highway(activation=highway, W_regularizer=keras.regularizers.l2(0.1))(feature)
#             else:
#                 feature = Highway(activation='relu', W_regularizer=keras.regularizers.l2(0.1))(feature)
#         else:
#             if nohighway:
#                 feature = Dense(units=600, activation=nohighway, kernel_regularizer=keras.regularizers.l2(0.1))(feature)
#             else:
#                 pass
        feature = Dense(600, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.1))(feature)
        feature = BatchNormalization()(feature)
        feature = Activation('relu')(feature)
    else:
        feature = Flatten()(char_embedding)
    # if position:
    #     word_feature_encoder = Model(inputs=[word_input, idc_input, pos_input], outputs=feature)
    # else:
    # print(word_input)
    # print(feature)
    word_feature_encoder = Model(word_input, feature)
    return word_feature_encoder

def build_model(radical_vocab_size=2487, word_vocab_size=10, char_vocab_size=10, max_sentence_length=500, max_word_length=4,
                classes=2, attention=False, dropout=0, char_emb_dim=15, comp_width=3,
                word=False, char=False, char_shape=True, model="rnn", cnn_encoder=True,
                highway=None, nohighway=None, shape_filter=True, char_filter=True, position=True):
    """
    build the rnn of words, use the output of build_word_feature as the feature of each word
    """
    print('build sentence rnn')
    if char_shape:
        word_feature_encoder = build_word_feature_shape(vocab_size=radical_vocab_size,
                                                        cnn_encoder=cnn_encoder, char_emb_dim=char_emb_dim, comp_width=comp_width,
                                                        highway=highway, nohighway=nohighway,
                                                        shape_filter=shape_filter,
                                                        char_filter=char_filter,
                                                        position=position)
        sentence_input = Input(shape=(max_sentence_length, 3, comp_width * max_word_length), dtype='int32')
        word_feature_sequence = TimeDistributed(word_feature_encoder)(sentence_input)
        # print(word_feature_sequence._keras_shape)
    if word:
        sentence_word_input = Input(shape=(max_sentence_length,), dtype='int32')
        word_embedding_sequence = Embedding(input_dim=word_vocab_size, output_dim=WORD_DIM)(sentence_word_input)
    if char:
        word_feature_encoder = build_word_feature_char(vocab_size=char_vocab_size,cnn_encoder=cnn_encoder, highway=highway)
        char_input = Input(shape=(max_sentence_length, max_word_length), dtype='int32')
        word_feature_sequence = TimeDistributed(word_feature_encoder)(char_input)
    if char_shape and word and not char:
        word_feature_sequence = Concatenate(axis=2)([word_feature_sequence, word_embedding_sequence])
    if word and not char_shape and not char:
        word_feature_sequence = word_embedding_sequence
    # print(word_feature_sequence._keras_shape)
    if model == "rnn":
        if attention:
            lstm_rnn = Bidirectional(LSTM(150, dropout=dropout, return_sequences=True))(word_feature_sequence)
            if highway:
                lstm_rnn = TimeDistributed(Highway(activation=highway))(lstm_rnn)
            elif nohighway:
                lstm_rnn = TimeDistributed(Dense(units=300, activation=nohighway))(lstm_rnn)
            lstm_rnn = AttentionWithContext()(lstm_rnn)
        else:
            lstm_rnn = Bidirectional(LSTM(150, dropout=dropout, return_sequences=False))(word_feature_sequence)
        x = lstm_rnn
    else:
        x = Flatten()(word_feature_sequence)
    if classes < 2:
        print("class number cannot less than 2")
        exit(1)
    else:
        preds = Dense(classes, activation='softmax')(x)
    if char_shape and not word and not char:
        sentence_model = Model(sentence_input, preds)
    if word and not char_shape and not char:
        sentence_model = Model(sentence_word_input, preds)
    if word and char_shape and not char:
        sentence_model = Model([sentence_input, sentence_word_input], preds)
    if char and not word and not char_shape:
        sentence_model = Model(char_input, preds)
    sentence_model.summary()
    return sentence_model
