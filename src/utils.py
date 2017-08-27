from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt

from odin import visual as V

from processing import LAUGH


def plot_laugh_prediction(y_pred, y_true, name):
    n = len(y_true)
    y = [y_true]
    for i in y_pred[:, 1:].T:
        y.append(i)
    y = [np.repeat(np.expand_dims(i, 0), repeats=n // 10, axis=0)
         for i in y]
    # ====== start ploting ====== #
    plt.figure()
    for i, x in enumerate(y):
        ax = plt.subplot(len(y), 1, i + 1)
        if i == 0:
            interpolation = 'nearest'
            title = "True Values"
        else:
            interpolation = 'bilinear'
            title = "Laugh type: %s" % LAUGH[i]
        ax.imshow(x, cmap='Reds', interpolation=interpolation,
                  alpha=0.9)
        ax.axis('off')
        ax.set_title(title)
    plt.tight_layout()
    plt.suptitle("File: %s" % name)


def plot_2D_threshold(threshold, performance):
    plt.figure()
    best_idx = np.argmax(performance)
    best_performance = performance[best_idx]
    best_threshold = threshold[best_idx]

    plt.plot(threshold, performance)
    plt.scatter(best_threshold, best_performance, s=30, c='r')
    plt.annotate(s="(p:%.2f F1:%.2f)" % (best_threshold, best_performance),
                 xy=(best_threshold - 0.15, best_performance + 0.005))
    plt.xlabel('Threshold probability value')
    plt.ylabel('F1 score on validation set')
    plt.suptitle('Threshold optimization')


def plot_3D_threshold(threshold, performance):
    # ====== best performance ====== #
    best_idx = np.argmax(performance)
    best_threshold = threshold[best_idx]
    best_performance = performance[best_idx]
    # ====== create image ====== #
    n = int(np.sqrt(len(threshold)))
    X = np.empty(shape=(n, n), dtype='float32')
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            X[i, j] = performance[idx]
            if best_performance == performance[idx]:
                row = i
                col = j
    plt.figure()
    ax = plt.subplot()
    ax.imshow(X[::-1], cmap='Reds', interpolation='nearest')
    ax.autoscale(False)
    ax.scatter(col + 1, n - row - 1, s=25, c='b')
    # ====== set axis ====== #
    idx = np.arange(0, n, n // 10)
    plt.xticks(idx, ['%.1f' % i for i in np.linspace(0, 1., n)[idx]])
    plt.yticks(n - idx - 1, ['%.1f' % i for i in np.linspace(0, 1., n)[idx]])
    # ====== labels ====== #
    plt.xlabel('Threshold probability for: fl')
    plt.ylabel('Thresholh probability for: st')
    plt.suptitle('Best threshold optimization\np(fl)=%.2f p(st)=%.2f F1=%.2f' %
        (best_threshold + (best_performance,)))


plot_save = V.plot_save
