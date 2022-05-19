#STANDARD PLOTTING FUNCTIONS

import sys
import ast
from copy import copy
import numpy as np
from numpy import arange

from scipy.spatial.distance import cdist,pdist
from scipy.stats import levene, pearsonr

from pandas import Series, DataFrame

from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def funcLinear(x, a, b):
    return a*x + b

def line(x, xpoints, ypoints):    
    return ((ypoints[1] - ypoints[0]) / (xpoints[1] - xpoints[0])) * (x - xpoints[0]) + ypoints[0]

def plot_gth_pre(Y_label, Y_pre, range_set = True, range_x=[-1.5,3.5], range_y=[-1.5,3.5], tag='Train', mod='model'):
    from scipy.optimize import curve_fit
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    plt.figure(figsize=(6,4))
    pre_Y = Y_pre
    parameter, covariance_matrix = curve_fit(funcLinear, Y_label.astype(float), pre_Y.flatten().astype(float))
#     parameter, covariance_matrix = curve_fit(funcLinear, Y_label.flatten().astype(float), pre_Y.flatten().astype(float))
    xx = np.linspace(min(Y_label)-0.1, max(Y_label)+.1, 30)
    plt.plot(xx, funcLinear(xx, *parameter), 'r--', label='fit')
    axes = plt.gca()
    if range_set:
        axes.set_xlim([-0.1,1.1])
        axes.set_ylim([-0.1,1.1])
    else:
        axes.set_xlim(range_x)
        axes.set_ylim(range_y)

    lims = [
        np.min([axes.get_xlim(), axes.get_ylim()]),  # min of both axes
        np.max([axes.get_xlim(), axes.get_ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ############# Prediction vs. Ground Truth ###################
    plt.scatter(pre_Y,Y_label,
                marker='o',
                s=80, linewidth=2,
                facecolors='none', 
                edgecolors='b')
    plt.legend(['Best Fit','Perfect Fit', 'Data'], loc='lower right')
    plt.xlabel('Standardised Predicted Value')
    plt.ylabel('Standardised True Value')
    plt.savefig('./fig/'+mod+tag+'.png', bbox_inches='tight', dpi=300)

    plt.show()
    
def pred_gtr(X_test, Y_test, model, range_set = True, tag='Train', mod='model',  range_x=[-1.5,3.5], range_y=[-1.5,3.5]):
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    from scipy import stats
    
    pre_Y = model.predict(X_test)
    plot_gth_pre(Y_test, pre_Y, range_set=range_set, tag=tag, mod=mod, range_x=range_x, range_y=range_y)
    
    
def plot_confusion_matrix_(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    plt.figure(figsize=(20,20)) 

    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
     
    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#def plot_embedding(X, label_Y, color_Y, discrete=True, title=None):
def plot_embedding(X, label_Y, color_Y, discrete=True, title=True):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(10,10))
    plt.grid(b=True)

    if discrete:
        plt.scatter(X[:, 0], X[:, 1],s = 180, alpha=0.8, edgecolors = 'black', c=color_Y, 
                    cmap=plt.cm.get_cmap("tab20", len(list(set(color_Y))) ))
        plt.colorbar( ticks= range(1, len(list(set(color_Y)))+1  )  )
    else:
        plt.scatter(X[:, 0], X[:, 1],s = 180, alpha=0.8, edgecolors = 'black', c=color_Y, 
                    cmap=plt.cm.get_cmap("jet"))
        plt.colorbar()
    
    plt.xlim([-.1,1.1])
    plt.ylim([-.1,1.1])
    plt.title(title)
    plt.axis('off')
    plt.xticks([]), plt.yticks([])

    
def plot_feature_importance(ft_set, feature_importance, show_cols = 30):
    
    fig = plt.figure(figsize=(12,4))
    w_lr_sort, ft_sorted, _ = return_feature_importance(ft_set, feature_importance, show_cols = show_cols)
    x_val = list(range(len(w_lr_sort)))

    plt.bar(x_val, w_lr_sort)
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


def plot_learning_curve(estimator, title, X, y, ylim=(0,1), cv=None,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    from sklearn.model_selection import learning_curve
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Instances")
    plt.ylabel("Mean Squard Error")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = -np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = -np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major')
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Train")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation")
    plt.legend(['Train','Cross-validation'], loc='upper right', fontsize=16)

#     plt.legend(loc="upper right")
    return plt



def kmeansElbow(x, **kwargs):
    # Reference:
    # http://stackoverflow.com/a/6657095
    filename = kwargs.get('filename', None)
    X_KM = x

    ##### cluster data into K=1..20 clusters #####
    K_MAX = 12
    KK = range(1,K_MAX+1)

    KM = [KMeans(n_clusters=k, random_state=0).fit(X_KM) for k in KK]
    centroids = [km.cluster_centers_ for km in KM]
    D_k = [cdist(X_KM, cent, 'euclidean') for cent in centroids]
    #cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]

    tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
    totss = sum(pdist(X_KM)**2)/X_KM.shape[0]  # The total sum of squares
    betweenss = totss - tot_withinss  # The between-cluster sum of squares

    ##### plots #####
    kIdx = 1  # Elbow
    #clr = cm.spectral( np.linspace(0,1,20) ).tolist()
    #mrk = 'os^p<dvh8>+x.'
    #variance_retain = betweenss/totss*100
    
    figsize=(8,5)

    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(KK, betweenss/totss*100, marker='o', markersize=6, markeredgewidth=1, markeredgecolor='b',)
    ax.plot(KK[kIdx], betweenss[kIdx]/totss*100, marker='o', markersize=0,
    markeredgewidth=0, markeredgecolor='r', markerfacecolor='None')
    ax.set_ylim((0,100))
    plt.grid(True)
    bbox_inches='tight'
    plt.xlabel('Number of clusters')
    plt.ylabel('Explained Variance (%)')
    #plt.title('Elbow for KMeans clustering')
    if filename:
        #plt.tight_layout()
        plt.subplots_adjust(left=0.2)
        plt.subplots_adjust(bottom=-0.2)
        plt.subplots_adjust(top=1)
        plt.savefig(filename)
    plt.show()