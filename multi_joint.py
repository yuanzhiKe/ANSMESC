import pickle
import os
import keras
import tensorflow as tf
from keras.layers import Dropout, BatchNormalization, Activation
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.models import Model
from keras.layers import Embedding, Input, AveragePooling1D, MaxPooling1D, Conv1D, concatenate, TimeDistributed, Bidirectional, LSTM, Dense, Flatten, GRU, Lambda, Concatenate, Reshape
from keras.legacy.layers import Highway
from attention import AttentionWithContext
from tqdm import tqdm


def initial_emb(vocab_size, emb_dim):
    zero_pad = np.zeros((1, emb_dim))
    limit = np.sqrt(1/(vocab_size + emb_dim))
    embedding_matrix =  np.random.uniform(-limit,limit,(vocab_size-1, emb_dim))
    w = np.vstack((zero_pad, embedding_matrix))
    return w

def build_model(radical_vocab_size=2487, word_vocab_size=35314, char_vocab_size=21294, max_sentence_length=500, max_word_length=4, classes=2, word_dim=600, comp_width=3, char=True, word=True, radical=True):
    """
    replicate Yin 16
    """
    assert(radical or word or char)
    input_list=[]
    embed_list=[]
    if radical:
        sip_input = Input(shape=(max_sentence_length, 3, comp_width * max_word_length,))
        shape_reshape = Lambda(lambda x: x[:, :, 0, :])(sip_input)  # use lambda to slice. not using lambda lead to errors
        shape_reshape = Reshape((max_sentence_length * comp_width * max_word_length,))(shape_reshape)
        shape_embedding_sequence = Embedding(input_dim=radical_vocab_size, output_dim=word_dim)(shape_reshape)
        shape_embedding = AveragePooling1D(pool_size=max_sentence_length * comp_width * max_word_length)(shape_embedding_sequence)
        shape_embedding = Flatten()(shape_embedding)
        input_list.append(sip_input)
        embed_list.append(shape_embedding)

    if word:
        word_input = Input(shape=(max_sentence_length,), dtype='int32')
        word_embedding_sequence = Embedding(input_dim=word_vocab_size, output_dim=word_dim)(word_input)
        word_embedding = AveragePooling1D(pool_size = max_sentence_length)(word_embedding_sequence)
        word_embedding = Flatten()(word_embedding)
        input_list.append(word_input)
        embed_list.append(word_embedding)

    if char:
        char_input = Input(shape=(max_sentence_length, max_word_length,))
        char_reshape = Reshape((max_sentence_length * max_word_length,))(char_input)
        char_embedding_sequence = Embedding(input_dim=char_vocab_size, output_dim=word_dim)(char_reshape)
        char_embedding = AveragePooling1D(pool_size=max_sentence_length * max_word_length)(char_embedding_sequence)
        char_embedding = Flatten()(char_embedding)
        input_list.append(char_input)
        embed_list.append(char_embedding)
    
    if len(embed_list) > 1:
        x = Concatenate()(embed_list)
    else:
        x = embed_list[0]

    if classes < 2:
        print("class number cannot less than 2")
        exit(1)
    else:
        preds = Dense(classes, activation='softmax')(x)
    if len(input_list) > 1:
        sentence_model = Model(input_list, preds)
    else:
        sentence_model = Model(input_list[0], preds)
    sentence_model.summary()
    return sentence_model
