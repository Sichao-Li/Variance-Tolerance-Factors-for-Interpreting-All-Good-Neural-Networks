import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.keras.callbacks import Callback


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0, weights=[]):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.weights = weights

    def on_epoch_end(self, epoch, logs={}):
        # print(feature_model.layers[1].get_weights())
        current = logs.get(self.monitor)
        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.weights.append(self.model.layers[1].get_weights())
            self.model.stop_training = True


class Feature_Linear(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Feature_Linear, self).__init__()

    def build(self, input_shape):

        if len(input_shape)==2:
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
                shape=(input_shape[0], input_shape[1]),
                # initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1.1, seed=42),
                initializer=tf.keras.initializers.Ones(),
                trainable=True,
                # constraint=lambda x: tf.clip_by_value(x, 0, 1)
            )

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs, **kwargs):
        return tf.multiply(inputs, self.w)


def make_feature_model(base_model, shape=(30)):

    inputs = tf.keras.Input(shape=shape)

    feature_model = tf.keras.Sequential(
        [
            Feature_Linear(),
        ]
    )
    feature_model._name = "feature_extractor"
    x = feature_model(inputs)
    outputs = base_model(x)
    feature_model = tf.keras.Model(inputs, outputs)
    feature_model.summary()
    return feature_model


X, y = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

labels = load_boston()['feature_names']

base_model = tf.keras.models.load_model('Boston_linear_base_model_1000.h5')

base_model.trainable = False

stop_call_back = EarlyStoppingByLossVal(monitor='loss', value=22.5, verbose=0)


def training_iteration():
    feature_model = make_feature_model(base_model, shape=(13))

    opt = tf.keras.optimizers.Adam()

    feature_model.compile(loss='mse', optimizer=opt, metrics=['mae'], run_eagerly=True)

    feature_model.fit(X_train, y_train, epochs=500, batch_size=10, verbose=1, validation_data=(X_test, y_test),
                      shuffle=True, callbacks=[stop_call_back])


while len(stop_call_back.weights) < 300:
    training_iteration()

arr = np.array(stop_call_back.weights)
arr = np.reshape(arr, (len(arr), np.shape(arr)[-1]))

np.save('./boston_linear_weights_output', arr)