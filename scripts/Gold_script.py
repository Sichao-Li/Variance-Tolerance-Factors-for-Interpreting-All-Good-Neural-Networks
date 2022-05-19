import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from sklearn import preprocessing
import pandas as pd
import numpy as np


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


# load data from csv file
def load_data(filename='data.csv'):
    X_df = pd.read_csv(filename)
    X_df = X_df.fillna(0)
    return X_df


X_df = load_data('Au_nanoparticle_dataset.csv')
X_df.describe()

name_prefix = 'bulk'

feature_set_bulk = ['N_bulk', 'Avg_bulk', 'BCN_5', 'BCN_6', 'BCN_7', 'BCN_8', 'BCN_9', 'BCN_10', 'BCN_11', 'BCN_12', 'BCN_13', 'BCN_14', 'BCN_15', 'BCN_16', 'BCN_17', 'FCC', 'HCP', 'ICOS', 'DECA', 'q6q6_B0', 'q6q6_B1', 'q6q6_B2', 'q6q6_B3', 'q6q6_B4', 'q6q6_B5', 'q6q6_B6', 'q6q6_B7', 'q6q6_B8', 'q6q6_B9', 'q6q6_B10', 'q6q6_B11', 'q6q6_B12', 'q6q6_B13', 'q6q6_B14', 'q6q6_B15']

X_features = X_df[feature_set_bulk]
feature_names = X_features.columns.values

y_multilabel = X_df.iloc[:,-1:]
label_names = y_multilabel.columns.values

# data processing

X_scaled = preprocessing.StandardScaler().fit_transform(X_features)
y_scaled = preprocessing.StandardScaler().fit_transform(y_multilabel)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)



base_model = tf.keras.models.load_model('./base_model_{}.h5'.format(name_prefix))

base_model.trainable = False


for i in range(50):

    stop_call_back = EarlyStoppingByLossVal(monitor='mae', value=0.025, verbose=0)

    feature_model = make_feature_model(base_model, shape=(35))

    opt = tf.keras.optimizers.Adam()

    feature_model.compile(loss='mse', optimizer=opt, metrics=['mae'], run_eagerly=True)

    history = feature_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1, validation_data=(X_test, y_test), shuffle=True, callbacks=[stop_call_back])


arr = np.array(stop_call_back.weights)
arr = np.reshape(arr, (len(arr), np.shape(arr)[-1]))

np.save('./gold_weights_output', arr)