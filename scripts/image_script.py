import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

num_classes = 2
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

delete_list = []


for i in range(len(y_train)):
    if y_train[i] != 3 and y_train[i] != 8:
        delete_list.append(i)

y_train = np.delete(y_train, delete_list)
x_train = np.delete(x_train, delete_list, axis=0)

delete_list_test = []

for i in range(len(y_test)):
    if y_test[i] != 3 and y_test[i] != 8:
        delete_list_test.append(i)

y_test = np.delete(y_test, delete_list_test)
x_test = np.delete(x_test, delete_list_test, axis=0)

# print(x_train[1])
# plt.imshow(x_train[3])
# plt.show()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train[y_train==3] = 0
y_train[y_train==8] = 1
y_test[y_test==3] = 0
y_test[y_test==8] = 1

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_uniform'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)



class Feature_Linear(tf.keras.layers.Layer):

    def __init__(self, units=(None, 32, 32), **kwargs):
        super(Feature_Linear, self).__init__()
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


batch_size = 128

epochs = 500

earlycp = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy', min_delta=0, patience=5, verbose=0,
        mode='auto', baseline=None, restore_best_weights=False
    )

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[earlycp])

# tf.keras.models.save_model(model, 'base_model_image_classifier.h5')
#
# base_model = tf.keras.models.load_model('base_model_image_classifier.h5')

model.trainable = False

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

stop_call_back = EarlyStoppingByLossVal(monitor='loss', value=0.0039, verbose=0)

def make_feature_model(base_model):

    inputs = tf.keras.Input(shape=(28, 28, 1))

    feature_model = tf.keras.Sequential(
        [
            Feature_Linear((28, 28)),
        ]
    )

    feature_model._name = "client1"
    x = feature_model(inputs)
    outputs = base_model(x)
    feature_model = tf.keras.Model(inputs, outputs)
    return feature_model


weights = []
for i in range(800):
    feature_model = make_feature_model(model)

    feature_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    earlycp = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0, patience=5, verbose=0,
        mode='auto', baseline=None, restore_best_weights=False
    )

    feature_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[stop_call_back])

    outputs = [layer.weights for layer in feature_model.layers]

    # print(np.shape(outputs[1]))
    image = outputs[1][-1]

    img = image.numpy()

    weights.append(img)


file = open("weights_constraints_800.txt", "wb")
# save array to the file
np.save(file, weights)
# close the file
file.close()
