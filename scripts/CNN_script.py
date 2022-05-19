import tensorflow as tf
import numpy as np
print(tf.__version__)


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


batch_size = 128

base_model = tf.keras.models.load_model('base_model_image_classifier.h5')

(x_train, y_train), (x_test, y_test), three_sum, eight_sum = data_set_classify_three_eight()

# Benchmark
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = base_model.evaluate(x_train, y_train, batch_size=batch_size)
print("test loss, test acc:", results)

# Baseline
array_aux = np.zeros(np.shape(x_train))
results = base_model.evaluate(array_aux, y_train, batch_size=batch_size)

# Individula contribution
single_contribution_set = []
for i in range(len(x_train[-1])):
    for j in range(len(x_train[-1])):
        aux_array = []
        array_aux = np.zeros(np.shape(x_train))
        array_aux[i][j] = x_train[i][j]
        results = base_model.evaluate(array_aux, y_train, batch_size=batch_size)

        aux_array.append(results[0])
        aux_array.append(results[1])
        single_contribution_set.append(aux_array)

# All contribution
# Set each feature to be 0 and calculate the drop of change as its contribution
# it is different from permutation method, providing permutation method results
contribution_set = []
for i in range(28):
    for j in range(28):
        aux_array = []
        array_aux = np.ones(np.shape(x_train))
        array_aux[:,i,j] = 0
        X_new = array_aux * x_train
        results_new = base_model.evaluate(X_new, y_train, batch_size=batch_size)
        aux_array.append(results_new[0])
        aux_array.append(results_new[1])
        contribution_set.append(aux_array)


# Individual contribution of each single feature
# Compare this with permutation method
# Can be used to calculate feature interaction
contribution_set = np.round(contribution_set, 8)
for i in range(len(contribution_set)):
    contribution_set[i] = abs(contribution_set[i][0]-0.00031589)


# load results from saved file
file = open('weights_constraints_800.txt', "rb")
# read the file to numpy array
arr1 = np.load(file)
# close the file
weights_mean = sum(arr1)/50
np.shape(weights_mean)

# We find the contribution and coefficient relationship through previous method (benchmark)
weights_output = np.zeros(np.shape(arr1))
X_data = []
y_data = []
for k in range(len(weights_output)):
    for j in range(28):
        for i in range(28):
    #         banchmark
            con_feature = contribution_set[i][0]
            array_aux = np.ones(np.shape(x_train))
            array_aux[k,i,j,0] = 1-weights_mean[i,j,0]

            X_new = array_aux * x_train
            results_new = base_model.evaluate(X_new, y_train, batch_size=batch_size)
            results_round = np.round(results_new[0],8)
    #         the loss change to banchmark
            con = abs(0.00031589-results_new[0])
            weights_output[k,i,j,0] = con/con_feature

file = open("weights_constraints_800_coef.txt", "wb")
# save array to the file
np.save(file, weights_output)
# close the file
file.close()