import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import shap

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
            # initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1.1, seed=42),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            # constraint=lambda x: tf.clip_by_value(x, 0, 1)
        )

    def call(self, inputs, **kwargs):
        return tf.multiply(inputs, self.w)


# file = open('weights_constraints.txt', "rb")
# # read the file to numpy array
# arr1 = np.load(file)
# # close the file
# print(sum(arr1)/5)
#
# img = sum(arr1)/50
#
# img = np.exp(-((img - 1) * (img - 1)))
#
#
# # img = np.reciprocal(abs(img - 1))
#
# # img[img<0.75]=0
#
# plt.imshow(img[:, :, 0])
# plt.show()
# save array to the file

# close the file




model = tf.keras.models.load_model('./Generated_models/my_model_ones.h5', custom_objects={'Feature_Linear': Feature_Linear})

model.summary()

outputs = [layer.weights for layer in model.layers]

# print(np.shape(outputs[1]))
image = outputs[1][-1]

img = image.numpy()

# img = 1 - img
# img = tf.math.sigmoid(img)

img[img<1] = 0

# print(img)
# image = np.abs(image)

plt.imshow(img[:, :, 0])
plt.show()


