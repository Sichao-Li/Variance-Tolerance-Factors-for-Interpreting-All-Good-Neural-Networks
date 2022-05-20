import lime
from lime import lime_tabular
from scipy.linalg import solve
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
# from Ag_data import import_data
from tensorflow.keras.callbacks import Callback
from copy import copy
import scipy.stats as ss
import seaborn as sns; sns.set()
sns.set(color_codes=True)
sns.set(font_scale=1.2)


def linear_calculation(weights):
    l = np.ones((13, 13)) 
    for i in range(12):
        for j in range(13):
            l[i][j] = weights[i][0][0][j]
    x = l - 1
    X = x[:-1, 1:]
    Y = -x[:-1,0]
    out = solve(X, Y)
    return out

# Check if the loss in current training epoch is satisfied the defined requirement
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
            
            
# Normolization Function e^-(x-1)^2
def FI_Normolization(weights):
    return [np.exp(-np.power((weight[0] - 1), 2)) for weight in weights]



def rank_importance(labels, feature_importance):
    importance_idx = ss.rankdata(feature_importance)
    return importance_idx
    
    
def plot_feature_importance(ft_set, feature_importance, show_cols = 30):
    
    fig = plt.figure(figsize=(12,4))
    w_lr_sort, ft_sorted, _ = return_feature_importance(ft_set, feature_importance, show_cols = show_cols)
    x_val = list(range(len(w_lr_sort)))

    plt.bar(x_val, w_lr_sort)
    plt.ylabel('Ranking')
    plt.xticks(x_val, ft_sorted, rotation='vertical')
    
    return fig

def plot_feature_importance_reverse(ft_set, feature_importance, show_cols = 30):
    
    fig = plt.figure(figsize=(12,4))
    w_lr_sort, ft_sorted, _ = return_feature_importance(ft_set, feature_importance, show_cols = show_cols)
    x_val = list(range(len(w_lr_sort)))

    plt.bar(x_val, w_lr_sort, color='r')
    plt.ylabel('Ranking')
    plt.xticks(x_val, ft_sorted, rotation='vertical')
    
    return fig

def return_feature_importance(ft_set, feature_importance, show_cols = 30):

    w_lr = copy(np.abs(feature_importance))
    w_lr = 100 * (w_lr / w_lr.max())
    sorted_index_pos = [index for index, num in sorted(enumerate(w_lr), key=lambda x: x[-1], 
                   reverse=True)]

    ft_sorted = []
    w_lr_sort = []
    for i, idx in enumerate(sorted_index_pos):
        if i > show_cols:
            break
        ft_sorted.append(ft_set[idx])
        w_lr_sort.append(w_lr[idx])

    return w_lr_sort, ft_sorted, sorted_index_pos


def reshape_save(weights, file_save=''):
    weights = np.reshape(out[0], (22,13))
    np.save(file_save, weights)