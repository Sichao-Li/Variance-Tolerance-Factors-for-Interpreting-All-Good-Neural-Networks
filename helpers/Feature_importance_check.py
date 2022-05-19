import tensorflow as tf
import Feature_Importance_Layer
from sklearn.datasets import load_boston
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from Ag_data import import_data
import Feature_Importance_Layer
tf.executing_eagerly()

import shap
import numpy as np


X, y = load_boston(return_X_y=True)

model = Feature_Importance_Layer.load_model_from_h5("base_model_boston.h5")
outputs = [layer.weights for layer in model.layers]
# print(outputs[1])
Feature_Importance_Layer.RF_matrix(X ,y)
# Feature_Importance_Layer.connection_weights(model.layers[0].weights[0], model.layers[1].weights[0], model.layers[2].weights[0])

# # select a set of background examples to take an expectation over
# background = X[np.random.choice(X.shape[0], 100, replace=False)]
# # explain predictions of the model on four images
# # e = shap.DeepExplainer(model)
# # ...or pass tensors directly
# e = shap.DeepExplainer(model, X)
# shap_values = e.shap_values(X)

# tf.Tensor(
# [[  49.578583   40.55274    24.612602   53.79119  -107.71542    74.749886
#     44.367393   28.799528   30.809717   42.80019    49.66586    36.699684
#     52.510036]], shape=(1, 13), dtype=float32)
#
# [47.70333  49.09881  49.687984 46.986687 47.398655 49.31354  47.96524
#   41.846077 38.54931  46.790176 46.079857 46.753376 48.050056]