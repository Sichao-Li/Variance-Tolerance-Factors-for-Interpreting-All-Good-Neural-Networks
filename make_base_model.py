import tensorflow as tf
from keras import layers
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

def make_base_model_boston():
    base_model = tf.keras.Sequential([
        tf.keras.layers.Dense(input_dim=13, units=1)
    ])
    return base_model


def make_base_model_breast():
    base_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_dim=30, activation='sigmoid',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))
    ])
    return base_model


def make_base_model_CNN(input_shape):
    base_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_uniform'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(2, activation="softmax"),
        ]
    )
    return base_model


def make_base_model_MPL():
    base_model = tf.keras.Sequential([

        tf.keras.layers.Dense(32, activation='relu'),

        tf.keras.layers.Dense(16),

        tf.keras.layers.Dense(units=1)
    ])
    return base_model


def make_base_model_RNN():
    NUM_WORDS = 2000  # only use top 1000 words
    INDEX_FROM = 3  # word index offset
    max_review_length = 200
    embedding_vecor_length = 32
    base_model = Sequential()
    base_model.add(Embedding(NUM_WORDS, embedding_vecor_length, input_length=max_review_length))
    base_model.add(LSTM(100))
    base_model.add(Dense(1, activation='sigmoid'))
    return base_model