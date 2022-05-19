import tensorflow as tf
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as  plt
import os


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


def make_base_model(input_dim=13):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(64, input_dim=input_dim, activation='relu'))

    model.add(tf.keras.layers.Dense(64, activation='relu'))

    model.add(tf.keras.layers.Dense(1))

    model.summary()

    return model


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


def load_model_from_h5(path_to_model='my_model_zeros.h5'):
    model = tf.keras.models.load_model(path_to_model, custom_objects={'Feature_Linear': Feature_Linear})
    return model


def connection_weights(input_hidden, hidden_layer, hidden_output):
    pre_products = tf.matmul(input_hidden, hidden_layer)
    hidden_output = tf.transpose(hidden_output)
    products = tf.multiply(pre_products, hidden_output)
    weights = tf.reduce_sum(products, 1)
    print("Connection weights: ", weights)
    return weights

def permutation_matrix(model, X, y):
    r = permutation_importance(model, X, y, n_repeats=100, random_state=0, scoring='neg_mean_squared_error')
    print("Permutation Importance: ", r['importances_mean'])
    return r['importances_mean']


def RF_matrix(X, y):
    RF_model = RandomForestRegressor(random_state=42, max_leaf_nodes=None, max_features=None,n_jobs=-1,
                                 criterion='mse',max_depth=10,
                                 min_samples_leaf=1,min_samples_split=2, n_estimators=200)
    # fit the model
    RF_model.fit(X, y)
    # get importance
    importance = RF_model.feature_importances_
    # summarize feature importance
    print('RF : ', importance)
    return RF_model, importance

# TO DO: Sharply value


def check_training_process(file_name = "ones_constraints.txt"):
    file = open(file_name, "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    #close the file
    print(arr1)

    l = [3, 6, 9, 30, 49]
    for i in range(10):
        print(arr1[i][-1])
        img = arr1[i][-1][:, :, 0] - 1
        img = np.abs(img)
        # img[img<.3]=0
        plt.imshow(img, cmap='cool')
        plt.show()
        # plt.savefig('./output_image/iteration_{}.png'.format(i))


def save_model_Checkpoint(checkpoint_path = "./tmp/training1/cp.ckpt"):
    checkpoint_path = checkpoint_path
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    return checkpoint_dir, cp_callback


def data_set_classify_three_eight():
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
    y_train[y_train == 3] = 0
    y_train[y_train == 8] = 1
    y_test[y_test == 3] = 0
    y_test[y_test == 8] = 1

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]

    return (x_train, y_train), (x_test, y_test), three_sum, eight_sum


def data_set_classify_one_seven():
    # Model / data parameters
    num_classes = 2
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    delete_list = []
    one_list = []
    seven_list = []
    for i in range(len(y_train)):
        if y_train[i] == 1:
            one_list.append(i)
        if y_train[i] == 7:
            seven_list.append(i)
        if y_train[i] != 1 and y_train[i] != 7:
            delete_list.append(i)

    one_sum = np.zeros((28, 28))
    seven_sum = np.zeros((28, 28))

    for i in range(len(one_list)):
        one_sum = np.add(one_sum, x_train[one_list[i]])

    for i in range(len(seven_list)):
        seven_sum = np.add(seven_sum, x_train[seven_list[i]])
    y_train = np.delete(y_train, delete_list)
    x_train = np.delete(x_train, delete_list, axis=0)

    delete_list_test = []

    for i in range(len(y_test)):
        if y_test[i] != 1 and y_test[i] != 7:
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
    y_train[y_train == 1] = 0
    y_train[y_train == 7] = 1
    y_test[y_test == 1] = 0
    y_test[y_test == 7] = 1

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]

    return (x_train, y_train), (x_test, y_test), one_sum, seven_sum

def data_set_classify_zero_four():
    # Model / data parameters
    num_classes = 2
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    delete_list = []
    one_list = []
    seven_list = []
    for i in range(len(y_train)):
        if y_train[i] == 0:
            one_list.append(i)
        if y_train[i] == 4:
            seven_list.append(i)
        if y_train[i] != 0 and y_train[i] != 4:
            delete_list.append(i)

    one_sum = np.zeros((28, 28))
    seven_sum = np.zeros((28, 28))

    for i in range(len(one_list)):
        one_sum = np.add(one_sum, x_train[one_list[i]])

    for i in range(len(seven_list)):
        seven_sum = np.add(seven_sum, x_train[seven_list[i]])
    y_train = np.delete(y_train, delete_list)
    x_train = np.delete(x_train, delete_list, axis=0)

    delete_list_test = []

    for i in range(len(y_test)):
        if y_test[i] != 0 and y_test[i] != 4:
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
    y_train[y_train == 0] = 0
    y_train[y_train == 4] = 1
    y_test[y_test == 0] = 0
    y_test[y_test == 4] = 1

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]

    return (x_train, y_train), (x_test, y_test), one_sum, seven_sum



