import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.callbacks import Callback
from keras.preprocessing import sequence
import keras
import numpy as np
import copy
from multiprocessing import Pool, Process
# fix random seed for reproducibility
numpy.random.seed(3)
# load the dataset but only keep the top n words, zero the rest
top_words = 2000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 200
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
import tensorflow as tf


class Feature_Linear(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        super(Feature_Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) == 2:
            self.w = self.add_weight(
                shape=(1, input_shape[-1]),
                # initializer=tf.keras.initializers.RandomUniform(minval=3, seed=42),
                initializer=tf.keras.initializers.RandomUniform(),
                # initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                # constraint=lambda x: tf.clip_by_value(x, -1, 1)
            )
        else:
            self.w = self.add_weight(
                shape=(2000, 1),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                # constraint=lambda x: tf.clip_by_value(x, 0, 1)
            )

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs, **kwargs):
        weights = tf.multiply(rnn_weights, self.w)
        return tf.matmul(inputs, weights)


class Onehot_Linear(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        # self.units = units
        super(Onehot_Linear, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs, **kwargs):
        return tf.one_hot(tf.cast(inputs, dtype=tf.int64), 2000)


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        # print(feature_model.layers[1].get_weights())
        current = logs.get(self.monitor)
        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


# keras.models.save_model(model, 'RNN_base_model.h5')
model = keras.models.load_model('RNN_base_model.h5')
rnn_weights = model.layers[0].weights[0]
rnn_weights = rnn_weights.numpy()
# X_train_encoded = numpy.zeros((25000, 200, 2000))
#
# for i in range(len(X_train)):
#     for j in range(len(X_train[0])):
#         X_train_encoded[i][j][X_train[i][j]] = 1
#
# X_test_encoded = numpy.zeros((25000, 200, 2000))
#
# for i in range(len(X_test)):
#     for j in range(len(X_test[0])):
#         X_test_encoded[i][j][X_test[i][j]] = 1

stop_call_back = EarlyStoppingByLossVal(monitor='loss', value=1.15, verbose=0)

weights_output = []


for i in range(200):
    print('--------------------------- Round {} ------------------------'.format(i))
    feature_model = Sequential()
    feature_model.add(keras.Input(shape=(200)))
    feature_model.add(Onehot_Linear())
    feature_model.add(Feature_Linear(32))
    feature_model.add(LSTM(100, trainable=False))
    feature_model.add(Dense(1, activation='sigmoid', trainable=False))
    feature_model.summary()

    feature_model.layers[2].set_weights(model.layers[1].get_weights())
    feature_model.layers[3].set_weights(model.layers[2].get_weights())

    feature_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    feature_model.fit(X_train.astype(int), y_train, epochs=50, batch_size=64, verbose=1, callbacks=[stop_call_back])
    print(np.shape(feature_model.layers[1].get_weights()[0]))
    weights_output.append(feature_model.layers[1].get_weights()[0])
    print('--------------------------- Round {} complete  ------------------------'.format(i))

# np.save('./RNN_weights_output_1', np.array(weights_output))


