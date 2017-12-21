# -*- coding: utf-8 -*-
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
from matplotlib import rc
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# rc('text', usetex=True)
matplotlib.use('Agg')

import matplotlib.mlab as mlab
from matplotlib import pyplot as plt

import os
import random
from six.moves import cPickle
from collections import defaultdict, OrderedDict

import numpy as np

from odin.utils import Progbar
from odin import visual as V
from odin import fuel as F
from odin.stats import freqcount
from odin.preprocessing.signal import loudness2intensity, mel2hz, mel_frequencies

from const import featpath, STEP_LENGTH, utt_id, SR, FMIN, FMAX
from processing import (get_dataset, GENDER, ALL_LAUGH)

save_path = '/tmp/tmp.pdf'
all_ltype = [i for i in ALL_LAUGH if i is not None]
laugh_colors = {laugh: color
                for laugh, color in zip(all_ltype,
                    V.generate_random_colors(n=len(all_ltype)))}
all_feats = ['bnf', 'spec', 'mspec', 'mfcc',
             'energy', 'pitch', 'f0']
popular_ltype = ['fl, b', 'fl, e', 'fl, m',
                 'st, b', 'st, e', 'st, m']
SEED = 5218
np.random.seed(SEED)
random.seed(SEED)


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

def get_nonzeros(x):
  return x[x != 0.]

# ===========================================================================
# Const
# ===========================================================================
# laugh, gender, utt_duration
ds = F.Dataset(featpath, read_only=True)
all_indices = list(ds['indices'].items())
np.random.shuffle(all_indices)
print(ds)
# ====== get length ====== #
# data, nb_classes = get_dataset(dsname=['est', 'fin', 'sam'],
#                                feats=all_feats,
#                                context=1, step=1,
#                                mode='all', ncpu=None,
#                                return_single_data=True)
# data.set_batch(batch_size=512, batch_mode='file', seed=None)

# ===========================================================================
# Calculate laugh pitch and f0, energy
# ===========================================================================
target_language = None # set this if you want specific corpus

laugh_pitch = defaultdict(lambda: defaultdict(list))
laugh_f0 = defaultdict(lambda: defaultdict(list))
laugh_energy = defaultdict(lambda: defaultdict(list))
laugh_loud = defaultdict(lambda: defaultdict(list))

max_f0 = 0
max_pitch = 0
max_energy = 0
max_loud = 0

min_f0 = np.finfo(np.float32).max
min_pitch = np.finfo(np.float32).max
min_energy = np.finfo(np.float32).max
min_loud = np.finfo(np.float32).max
# ====== get all data ====== #
for name, (start, end) in all_indices:
  lang = name.split('/')[0]
  # check target language
  if target_language is not None:
    if lang != str(target_language):
      continue
  for s, e, laugh, spk in ds['laugh'][name]:
    if laugh not in popular_ltype: # only for popular
      continue
    gender = ds['gender'][lang][spk]
    # pitch
    pitch = get_nonzeros(ds['pitch'][start:end][s:e][:, 0])
    if len(pitch) > 0:
      max_pitch = max(max_pitch, max(pitch))
      min_pitch = min(min_pitch, min(pitch))
    # f0
    f0 = get_nonzeros(ds['f0'][start:end][s:e][:, 0])
    if len(f0) > 0:
      max_f0 = max(max_f0, max(f0))
      min_f0 = min(min_f0, min(f0))
    # loud
    loud = get_nonzeros(ds['loudness'][start:end][s:e][:, 0])
    loud = loud[loud > 0.12]
    if len(loud) > 0:
      max_loud = max(max_loud, max(loud))
      min_loud = min(min_loud, min(loud))
    # energy
    energy = loudness2intensity(loud)
    if len(energy) > 0:
      max_energy = max(max_energy, max(energy))
      min_energy = min(min_energy, min(energy))
    # save info
    laugh_pitch[gender][laugh] += pitch.tolist()
    laugh_f0[gender][laugh] += f0.tolist()
    laugh_energy[gender][laugh] += energy.tolist()
    laugh_loud[gender][laugh] += loud.tolist()
# ====== ploting helper====== #
def plotting(data, title, ymin, ymax, step=80, outlier=(0, 0)):
  """ outlier: tuple (min, max) """
  ymin = 0 if ymin < 50 else 50
  plt.figure(figsize=(len(popular_ltype), 3))
  n = 1
  data = sorted(data.items(), key=lambda x: x[0])
  #
  for gender, _ in data:
    color = '#1F77B4' if gender == 'F' else 'forestgreen'
    for laugh, x in sorted(_.items(), key=lambda x: x[0]):
      if len(x) == 0:
        n += 1
        continue
      min_outlier = np.percentile(x, q=outlier[0])
      max_outlier = np.percentile(x, q=100 - outlier[1])
      x = [i for i in x if min_outlier <= i <= max_outlier]
      plt.subplot(2, len(popular_ltype), n)
      mu, sigma = np.mean(x), np.std(x)
      _, bins, patches = plt.hist(x, bins=25, normed=1, orientation='horizontal',
                                  color=color, rwidth=0.88)
      # add a 'best fit' line
      y = mlab.normpdf(bins, mu, sigma)
      plt.plot(y, bins, 'r-.', linewidth=1)
      # mean line
      plt.hlines(y=mu, xmin=np.min(_), xmax=np.max(_),
                 linewidth=1, color='darkorange', linestyles='--',
                 label='μ=%d\nσ=%d' % (int(mu), int(sigma)))
      # tick information
      plt.xticks([], [], fontsize=6)
      plt.ylim(ymin, ymax)
      if n in (1, 7):
        np.linspace(ymin, ymax, num=6).astype(int)
        plt.yticks(np.arange(ymin, ymax, step).astype(int),
                   fontsize=6)
        plt.ylabel("Female" if gender == 'F' else 'Male')
      else:
        plt.yticks([], [], fontsize=6)
      # title
      if n < 7:
        plt.title(laugh, fontsize=8)
      n += 1
      # legend
      plt.legend(fontsize=5, loc=1)
  # main title
  plt.suptitle(title, fontsize=12)
  plt.subplots_adjust(wspace=0.02, hspace=0.08)
# ====== plot ====== #
print("Plotting pitch")
plotting(laugh_pitch, "Pitch", min_pitch, max_pitch)
print("Plotting F0")
plotting(laugh_f0, "F0", min_f0, max_f0)
print("Plotting Intensity")
plotting(laugh_energy, "Intensity", min_energy, max_energy,
        step=30)
print("Plotting Loudness")
plotting(laugh_loud, "Loudness", min_loud, max_loud)
# ===========================================================================
# Spectrogram
# ===========================================================================
def norm_0_1(x):
  mi = np.min(x, -1, keepdims=True)
  ma = np.max(x, -1, keepdims=True)
  return (x - mi) / (ma - mi)

laugh_spec = defaultdict(lambda: defaultdict(list))
max_spec = np.finfo(np.float32).min
min_spec = np.finfo(np.float32).max
samples1 = {}
samples2 = {}
# ====== get all data ====== #
for name, (start, end) in all_indices:
  lang = name.split('/')[0]
  for s, e, laugh, spk in ds['laugh'][name]:
    if laugh not in popular_ltype: # only for popular
      continue
    # get gender
    gender = ds['gender'][lang][spk]
    # get spectrogram
    spec = ds['mspec'][start:end]
    spec1 = spec[:, :40][s:e]
    spec2 = spec[:, 40:80][s:e]
    # save some samples
    sample_name = gender + '/' + laugh
    if sample_name not in samples1 and \
    np.random.rand() > 0.08 and \
    80 < spec1.shape[0] < 120:
      samples1[sample_name] = spec1
      samples2[sample_name] = spec2
    # laugh_spec
    if spec1.shape[0] > 0:
      spec = norm_0_1(spec[:, :40])
      spec = spec[s:e]
      laugh_spec[gender][laugh].append(spec)
    # store min, max
    max_spec = max(max_loud, np.max(spec))
    min_spec = min(min_loud, np.min(spec))
# ====== simple check ====== #
assert len(samples1) == len(popular_ltype) * 2
assert len(samples2) == len(popular_ltype) * 2
# ====== plot sample first ====== #
for samples, figname in zip([samples1, samples2],
                            ['Mel-scaled spectrogram', 'Delta mel-spectrogram']):
  plt.figure(figsize=(6, 2))
  samples = sorted(samples.items(), key=lambda x: x[0])
  for i, (name, spec) in enumerate(samples):
    gender, laugh = name.split('/')
    plt.subplot(2, len(popular_ltype), i + 1)
    V.plot_spectrogram(spec.T)
    if i in (0, 6):
      plt.ylabel('Female' if gender == 'F' else 'Male', fontsize=10)
    plt.title(laugh, fontsize=10)
  plt.suptitle(figname, fontsize=12)
  plt.subplots_adjust(wspace=0.04, hspace=0.06)
# ====== spectrogram at different frequencies ====== #
plt.figure(figsize=(15, 5))
n = 1
hz = np.round(mel_frequencies(n_mels=40, fmin=FMIN, fmax=FMAX))
hz = hz.astype(np.int32)

for gender, _ in [(gen, laugh_spec[gen]) for gen in ['F', 'M']]:
  for laugh, x in sorted(_.items(), key=lambda x: x[0]):
    x = np.concatenate(x, axis=0)
    x = np.sum(x, axis=0)
    x = (x - x.min()) / (x.max() - x.min())
    print(gender, laugh, x.shape)
    # plotting
    plt.subplot(2, len(popular_ltype), n)
    plt.plot(hz, x)
    if n in (1, 7):
      plt.ylabel('Female' if gender == 'F' else 'Male', fontsize=12)
      plt.ylim((0 - 0.02, 1 + 0.02))
      plt.yticks(np.linspace(0, 1, num=5))
    else:
      plt.yticks([])
    # x ticks information
    plt.xlim((min(hz), max(hz)))
    if n >= 7:
      plt.xticks([j for i, j in enumerate(hz)
                  if i == 0 or i == len(hz) - 1 or ((i + 1) % 8) == 0],
                 rotation=300, fontsize=8)
      if n == 7:
        plt.xlabel('Hz', fontsize=12)
    else:
      plt.title(laugh, fontsize=12)
      plt.xticks([])
    # increase counter
    n += 1
plt.suptitle("Mel-frequency contents", fontsize=14)
plt.subplots_adjust(wspace=0.09, hspace=0.06)
# ===========================================================================
# PCA
# ===========================================================================
from sklearn.manifold import TSNE
FEAT = 'mspec'
n_components = None
# ====== fast check the dataset ====== #
# f = all_files[12]
# start, end = ds['indices'][f]
# n = end - start
# laugh = np.ones(shape=(n, n // 8), dtype='float32')
# for s, e, lt, spk in ds['laugh'][f]:
#   if lt in all_ltype:
#     laugh[s:e, :] = 0.
# feat = {name: transform1(name, ds[name][start:end], 200, 800)
#         for name in all_feats}
# feat['Laughter'] = laugh[200:800]

# V.plot_features(feat, sharex=True)
V.plot_save('/tmp/tmp.pdf')
