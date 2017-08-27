from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt

from odin import visual as V

from processing import LAUGH


def plot_laugh_prediction(y_pred, y_true, f_calibrate, name):
    n = len(y_true)
    width = n // 10
    nb_classes = y_pred.shape[1]
    nb_row = 4 if nb_classes == 2 else 5

    # ====== helper ====== #
    def to_spectrogram(x):
        if nb_classes == 2:
            threshold = 0.2
        else:
            threshold = 1. / nb_classes
        x = np.where(x < threshold, 0., x)
        return np.repeat(x[None, :], repeats=width, axis=0)

    def to_image(vals):
        img = np.ones(shape=(width, n, 3))
        for i, y in enumerate(vals):
            if y == 1: # fl laugh
                img[:, i, :] = (1, 0, 0)
            elif y == 2: # st laugh
                img[:, i, :] = (0, 0, 1)
            else:
                img[:, i, :] = (0.8, 0.8, 0.8)
        return img

    def show(ax, img, interp, name, cmap=None):
        ax.imshow(img, cmap=cmap, interpolation=interp)
        ax.set_ylabel(name)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_yticklabels([]); ax.set_xticklabels([])
    # ====== add y_pred ====== #
    z_spec1 = to_spectrogram(y_pred[:, 1]) # probability spectrogram
    z_spec2 = to_spectrogram(y_pred[:, 2]) if nb_classes == 3 else None
    z_true = to_image(y_true)
    z_pred = to_image(np.argmax(y_pred, -1))
    z_calib = to_image(f_calibrate(y_pred))

    # ====== start ploting ====== #
    plt.figure()
    show(plt.subplot(nb_row, 1, 1), z_true, 'nearest', 'True')
    show(plt.subplot(nb_row, 1, 2), z_pred, 'nearest', 'Predict')
    show(plt.subplot(nb_row, 1, 3), z_calib, 'nearest', 'Calibrate')
    if nb_classes == 2:
        show(plt.subplot(nb_row, 1, 4), z_spec1, 'bilinear', 'p', 'Blues')
    else:
        show(plt.subplot(nb_row, 1, 4), z_spec1, 'bilinear', 'p(fl)', 'Blues')
        show(plt.subplot(nb_row, 1, 5), z_spec2, 'bilinear', 'p(st)', 'Reds')
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
