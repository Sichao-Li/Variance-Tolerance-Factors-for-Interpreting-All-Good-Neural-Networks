from helpers import utilities
import tensorflow as tf
from sklearn.model_selection  import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from keras.preprocessing import sequence
import pandas as pd
import numpy as np


def load_data_boston():
    X, y = utilities.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    labels = utilities.load_boston()['feature_names']
    return X_train, X_test, y_train, y_test, labels


def load_data_breast():
    X, y = load_breast_cancer(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    labels = load_breast_cancer()['feature_names']
    return X_train, X_test, y_train, y_test, labels


def load_data_CNN():
    # Model / data parameters
    num_classes = 2
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    delete_list = []
    three_list = []
    eight_list = []
    for i in range(len(y_train)):
        if y_train[i] == 3:
            three_list.append(i)
        if y_train[i] == 8:
            eight_list.append(i)
        if y_train[i] != 3 and y_train[i] != 8:
            delete_list.append(i)

    three_sum = np.zeros((28, 28))
    eight_sum = np.zeros((28, 28))

    for i in range(len(three_list)):
        three_sum = np.add(three_sum, x_train[three_list[i]])

    for i in range(len(eight_list)):
        eight_sum = np.add(eight_sum, x_train[eight_list[i]])
    y_train = np.delete(y_train, delete_list)
    x_train = np.delete(x_train, delete_list, axis=0)

    delete_list_test = []

    for i in range(len(y_test)):
        if y_test[i] != 3 and y_test[i] != 8:
            delete_list_test.append(i)

    y_test = np.delete(y_test, delete_list_test)
    x_test = np.delete(x_test, delete_list_test, axis=0)

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train[y_train == 3] = 0
    y_train[y_train == 8] = 1
    y_test[y_test == 3] = 0
    y_test[y_test == 8] = 1

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    return (x_train, y_train), (x_test, y_test)

def load_data_MPL(set_name):
    X_df = pd.read_csv('Gold_dataset/Au_nanoparticle_dataset.csv')
    X_df = X_df.fillna(0)
    X_features = X_df[set_name]
    feature_names = X_features.columns.values

    y_multilabel = X_df.iloc[:,-1:]
    label_names = y_multilabel.columns.values
    X_scaled = preprocessing.StandardScaler().fit_transform(X_features)
    y_scaled = preprocessing.StandardScaler().fit_transform(y_multilabel)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def load_data_RNN():
    NUM_WORDS = 2000  # only use top 1000 words
    INDEX_FROM = 3  # word index offset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
    # truncate and pad input sequences
    max_review_length = 200
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    return X_train, X_test, y_train, y_test

