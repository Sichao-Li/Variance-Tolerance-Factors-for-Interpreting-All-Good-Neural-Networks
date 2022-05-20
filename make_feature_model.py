import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


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


class Feature_Linear_CNN(tf.keras.layers.Layer):

    def __init__(self, units=(None, 32, 32), **kwargs):
        super(Feature_Linear_CNN, self).__init__()
        self.units = units

    def get_config(self):
        config = super().get_config()
        config["units"] = self.units
        return config

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(28, 28, 1),
            # initializer='random_normal',
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            # constraint=lambda x: tf.clip_by_value(x, 0, 1),
            name='weights1',
        )

    def call(self, inputs, **kwargs):
        output = tf.multiply(inputs, self.w)
        return output


def make_feature_model_CNN(base_model):

    inputs = tf.keras.Input(shape=(28, 28, 1))

    feature_model = tf.keras.Sequential(
        [
            Feature_Linear_CNN((28, 28)),
        ]
    )

    feature_model._name = "client1"
    x = feature_model(inputs)
    outputs = base_model(x)
    feature_model = tf.keras.Model(inputs, outputs)
    return feature_model


class Feature_Linear_RNN(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        super(Feature_Linear_RNN, self).__init__(**kwargs)

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

    def call(self, inputs, base_model, **kwargs):
        rnn_weights = base_model.layers[0].weights[0]
        rnn_weights = rnn_weights.numpy()
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


def make_feature_model_RNN(base_model):
    feature_model = Sequential()
    feature_model.add(tf.keras.Input(shape=(200)))
    feature_model.add(Onehot_Linear())
    feature_model.add(Feature_Linear_RNN(32))
    feature_model.add(LSTM(100, trainable=False))
    feature_model.add(Dense(1, activation='sigmoid', trainable=False))
    feature_model.layers[2].set_weights(base_model.layers[1].get_weights())
    feature_model.layers[3].set_weights(base_model.layers[2].get_weights())
    print(feature_model.summary())
    return feature_model

