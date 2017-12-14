# ===========================================================================
# Plot the laugh duration hitogram compared among 3 datasets
# Layout:
# | GeneralLaugh-Sami | Estonian | Finnish | (histogram)
# |    FL   |   ST   |          |         | (boxplot)
# ----------------------------------------
# Plotting the time - laughter occurences matrices
# Emotional states and laughter types correlation matrix
########################################################
# Acoustic analysis
# Show sampled spectrogram, of different laughter types, from different
# gender in different dataset (plot with the pitch and energy values).
# Laughter type Occurences vs gender
# Laughter types + emotional states Occurences vs gender
# Laughter type F0 vs gender
# Laughter type + emotional states F0 vs gender
# ===========================================================================
from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
from six.moves import cPickle
from collections import defaultdict, OrderedDict

import numpy as np

from odin.utils import Progbar
from odin import visual as V
from odin import fuel as F
from odin.stats import freqcount

from const import featpath, STEP_LENGTH, utt_id
from processing import (get_dataset, GENDER, ALL_LAUGH)

save_path = '/tmp/tmp.pdf'
all_ltype = [i for i in ALL_LAUGH if i is not None]
laugh_colors = {laugh: color
                for laugh, color in zip(all_ltype,
                    V.generate_random_colors(n=len(all_ltype)))}
all_feats = ['dbf', 'spec', 'mspec', 'mfcc',
             'energy', 'pitch', 'f0']
popular_ltype = ['fl, b', 'fl, e', 'fl, m',
                 'st, b', 'st, e', 'st, m']


def timeFMT(dur):
  min = int(dur)
  frac = dur - min
  if frac > 0:
    frac = int(frac * 60)
    return str(min) + ':%.02d' % frac
  else:
    return str(min) + ':00'


# ===========================================================================
# Helper
# ===========================================================================
def alias2name(lang):
  return 'Sami' if lang == 'sam' else \
      ('Finnish' if lang == 'fin' else 'Estonian')


def transform1(name, X, start, end):
  if name in ('mspec', 'mfcc', 'f0', 'pitch', 'energy'):
    n = X.shape[1]
    X = X[:, :n // 3]
  X = X[int(start):int(end)]
  return X

# ===========================================================================
# Const
# ===========================================================================
# laugh, gender, utt_duration
ds = F.Dataset(featpath, read_only=True)
all_files = sorted(list(ds['indices'].keys()))
# ====== get length ====== #
# data, nb_classes = get_dataset(dsname=['est', 'fin', 'sam'],
#                                feats=all_feats,
#                                context=1, step=1,
#                                mode='all', ncpu=None,
#                                return_single_data=True)
# data.set_batch(batch_size=512, batch_mode='file', seed=None)

# ====== fast check the dataset ====== #
f = all_files[12]
start, end = ds['indices'][f]
n = end - start
laugh = np.ones(shape=(n, n // 8), dtype='float32')
for s, e, lt, spk in ds['laugh'][f]:
  if lt in all_ltype:
    laugh[s:e, :] = 0.
feat = {name: transform1(name, ds[name][start:end], 200, 800)
        for name in all_feats}
feat['Laughter'] = laugh[200:800]

V.plot_features(feat)
V.plot_save('/tmp/tmp.pdf')
