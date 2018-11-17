from __future__ import print_function, division, absolute_import

import os
from collections import OrderedDict
from six.moves import cPickle

import numpy as np

from odin.utils import get_all_files, ArgController

args = ArgController(
).add('--person', 'include person bounding box', False
).add('--legs', 'include legs bounding box', False
).add('-stack', 'stack multiple frame into one', 1
).add('-shift', 'the shift distance between each frame in stacking', 1
).add('-delta', 'the order delta to be compute', 2
).parse()

from odin.preprocessing.speech import compute_delta

# ===========================================================================
# person, head, body
# Load data
# The format of the data in the CSV files is as follows:
# Column 1:        video frame number
# Column 2-5:      left person bounding box
# Column 6-9:      right person bounding box
# Column 10-13:    left head bounding box
# Column 14-17:    right head bounding box
# Column 18-21:    left body bounding box
# Column 22-25:    right body bounding box
# Column 26-29:    left legs bounding box
# Column 30-33:    right legs bounding box
# Each bounding box consists of 4 numbers (x, y, w, h) as follows:
# x:    x-coordinate of top left corner of bounding box (distance from left edge of frame, in pixels)
# y:    y-coordinate of top left corner of bounding box (distance from top edge of frame, in pixels)
# w:   width of bounding box (in pixels)
# h:   height of bounding box (in pixels)
# ===========================================================================
path = '/Users/trungnt13/OneDrive - University of Helsinki/data/EstonianFirstEncounters/boundingbox'
outpath = '/Users/trungnt13/tmp/estonia_box'
files = get_all_files(path, filter_func = lambda x: '.csv' in x)
data = OrderedDict()


def stack(x, n, shift):
    idx = list(range(0, x.shape[0], shift))
    _ = [x[i:i + n].ravel() for i in idx
         if (i + n) <= x.shape[0]]
    x = np.asarray(_) if len(_) > 1 else _[0]
    return x

# ===========================================================================
# Load data
# ===========================================================================
for f in files:
    X = np.genfromtxt(f, delimiter=',', dtype='int32')[:, 1:]
    # order: person, head, body, legs
    X_left = [X[:, 8:12], X[:, 16:20]]
    X_right = [X[:, 12:16], X[:, 20:24]]
    if args['person']: # include person bounding box
        X_left = [X[:, 0:4]] + X_left
        X_right = [X[:, 4:8]] + X_right
    if args['legs']: # include legs bounding box
        X_left = X_left + [X[:, 24:28]]
        X_right = X_right + [X[:, 28:32]]
    X_left = np.concatenate(X_left, axis=-1)
    X_right = np.concatenate(X_right, axis=-1)
    # ====== stacking ====== #
    if args['stack'] > 1:
        shift = shift = args['stack'] if args['shift'] <= 0 else int(args['shift'])
        X_left = stack(X_left, args['stack'], shift)
        X_right = stack(X_right, args['stack'], shift)
    # ====== compute delta ====== #
    if args['delta'] > 0:
        X_left = compute_delta(X_left, width=9, order=args['delta'],
                               axis=0, trim=True)
        X_right = compute_delta(X_right, width=9, order=args['delta'],
                               axis=0, trim=True)
        if args['delta'] == 1:
            X_left = X_left[0]
            X_right = X_right[0]
        else:
            X_left = np.concatenate(X_left, axis=-1)
            X_right = np.concatenate(X_right, axis=-1)
    # ====== dtype ====== #
    X_left = X_left.astype('float32')
    X_right = X_right.astype('float32')
    name = os.path.basename(f).replace('.csv', '')
    print(name, ':', X_left.shape, X_right.shape)
    data[name] = (X_left, X_right)

# ===========================================================================
# Save data
# ===========================================================================
print("Saving data to disk ...")
cPickle.dump(data,
            open(outpath, 'w'),
            protocol=cPickle.HIGHEST_PROTOCOL)

# ===========================================================================
# Visualization for testing
# ===========================================================================
# import matplotlib
# matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
# from odin import visual
# n = X_left.shape[-1]
# ax = plt.subplot(2, 1, 1)
# visual.plot_spectrogram(X_left.T[:n // 2, :500], ax=ax)
# ax = plt.subplot(2, 1, 2)
# visual.plot_spectrogram(X_left.T[n // 2:, :500], ax=ax)
# visual.plot_show(True, True)
