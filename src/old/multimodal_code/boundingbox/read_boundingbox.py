from __future__ import print_function, division, absolute_import

import os
from collections import OrderedDict
from six.moves import cPickle

import numpy as np
from scipy import signal

from odin.utils import get_all_files, ArgController

args = ArgController(
).add('--person', 'include person bounding box', False
).add('--legs', 'include legs bounding box', False
).add('-stack', 'stack multiple frame into one', 1
).add('-shift', 'the shift distance between each frame in stacking', 1
).add('-delta', 'the order delta to be compute', 2
).parse()

# ===========================================================================
# Compute delta coefficients features
# ===========================================================================
def compute_delta(data, width=9, order=1, axis=0):
  r'''Compute delta features: local estimate of the derivative
  of the input data along the selected axis.
  Original implementation: librosa

  Parameters
  ----------
  data      : np.ndarray
      the input data matrix (e.g. for spectrogram, delta should be applied
      on time-axis).
  width     : int >= 3, odd [scalar]
      Number of frames over which to compute the delta feature
  order     : int > 0 [scalar]
      the order of the difference operator.
      1 for first derivative, 2 for second, etc.
  axis      : int [scalar]
      the axis along which to compute deltas.
      Default is -1 (columns).

  Returns
  -------
  delta_data   : list(np.ndarray) [shape=(d, t) or (d, t + window)]
      delta matrix of `data`.
      return list of deltas

  Examples
  --------
  Compute MFCC deltas, delta-deltas
  >>> mfcc = mfcc(y=y, sr=sr)
  >>> mfcc_delta1, mfcc_delta2 = compute_delta(mfcc, 2)
  '''
  data = np.atleast_1d(data)

  if width < 3 or np.mod(width, 2) != 1:
    raise ValueError('width must be an odd integer >= 3')

  order = int(order)
  if order <= 0:
    raise ValueError('order must be a positive integer')

  half_length = 1 + int(width // 2)
  window = np.arange(half_length - 1., -half_length, -1.)

  # Normalize the window so we're scale-invariant
  window /= np.sum(np.abs(window)**2)

  # Pad out the data by repeating the border values (delta=0)
  padding = [(0, 0)] * data.ndim
  width = int(width)
  padding[axis] = (width, width)
  delta_x = np.pad(data, padding, mode='edge')

  # ====== compute deltas ====== #
  all_deltas = []
  for _ in range(order):
    delta_x = signal.lfilter(window, 1, delta_x, axis=axis)
    all_deltas.append(delta_x)
  # ====== Cut back to the original shape of the input data ====== #
  trim_deltas = []
  for delta_x in all_deltas:
    idx = [slice(None)] * delta_x.ndim
    idx[axis] = slice(- half_length - data.shape[axis], - half_length)
    delta_x = delta_x[idx]
    trim_deltas.append(delta_x.astype('float32'))
  return trim_deltas[0] if order == 1 else trim_deltas

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
