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
from six.moves import cPickle
from collections import defaultdict, OrderedDict

import numpy as np

from odin.utils import Progbar, UnitTimer
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
all_indices = sorted(ds['indices'].items(), key=lambda x: x[0])
np.random.shuffle(all_indices)
print(ds)
# ====== get length ====== #
# data, nb_classes = get_dataset(dsname=['est', 'fin', 'sam'],
#                                feats=all_feats,
#                                context=1, step=1,
#                                mode='all', ncpu=None,
#                                return_single_data=True)
# data.set_batch(batch_size=512, batch_mode='file', seed=None)
def iterate_laughter(feat, target_language=None):
  """ lang, laugh, spk, gender, feat """
  for name, (start, end) in all_indices:
    lang = name.split('/')[0]
    # check target language
    if target_language is not None:
      if lang != str(target_language):
        continue
    # iterate
    for s, e, laugh, spk in ds['laugh'][name]:
      if laugh not in popular_ltype: # only for popular
        continue
      # get gender
      gender = ds['gender'][lang][spk]
      if isinstance(feat, (tuple, list)):
        x = [ds[name][start:end][s:e] for name in feat]
      else:
        x = ds[feat][start:end][s:e]
      yield lang, laugh, spk, gender, x
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
for lang, laugh, spk, gender, (pitch, f0, loud) in iterate_laughter(
    feat=('pitch', 'f0', 'loudness'), target_language=None):
  # pitch
  pitch = get_nonzeros(pitch[:, 0])
  if len(pitch) > 0:
    max_pitch = max(max_pitch, max(pitch))
    min_pitch = min(min_pitch, min(pitch))
  # f0
  f0 = get_nonzeros(f0[:, 0])
  if len(f0) > 0:
    max_f0 = max(max_f0, max(f0))
    min_f0 = min(min_f0, min(f0))
  # loud
  loud = get_nonzeros(loud[:, 0])
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
samples1 = defaultdict(list)
samples2 = defaultdict(list)
# ====== get all data ====== #
for lang, laugh, spk, gender, spec in iterate_laughter('mspec'):
  # get spectrogram
  spec1 = spec[:, :40]
  spec2 = spec[:, 40:80]
  # save some samples
  sample_name = gender + '/' + laugh
  if len(samples1[sample_name]) < 2 and \
  np.random.rand() > 0.08 and \
  80 < spec1.shape[0] < 130:
    samples1[sample_name].append(spec1)
    samples2[sample_name].append(spec2)
  # laugh_spec
  if spec1.shape[0] > 0:
    spec = norm_0_1(spec[:, :40])
    laugh_spec[gender][laugh].append(spec)
  # store min, max
  max_spec = max(max_loud, np.max(spec))
  min_spec = min(min_loud, np.min(spec))
# ====== simple check ====== #
for n, s in sorted(samples1.items(), key=lambda x: x[0]):
  assert len(s) == 2
# ====== plot samples first ====== #
for samples, figname in zip([samples1, samples2],
                            ['Mel-scaled spectrogram', 'Delta mel-spectrogram']):
  plt.figure(figsize=(6, 4))
  for idx, (name, (spec1, spec2)) in enumerate(
      sorted(samples.items(), key=lambda x: x[0])):
    gender, laugh = name.split('/')
    row = 1 if gender == 'F' else len(popular_ltype) * 2 + 1
    idx = row + idx % len(popular_ltype)
    # first row
    plt.subplot(4, len(popular_ltype), idx)
    V.plot_spectrogram(spec1.T)
    if idx in (1, len(popular_ltype) * 2 + 1):
      plt.ylabel('Female' if gender == 'F' else 'Male', fontsize=10)
    # only title for first row
    if gender == 'F':
      plt.title(laugh, fontsize=10)
    # second row
    plt.subplot(4, len(popular_ltype), idx + len(popular_ltype))
    V.plot_spectrogram(spec2.T)
  # main title
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
# ====== color and marker ====== #
styles = {gender + '/' + laugh: (color, marker)
  for laugh, color in zip(popular_ltype,
                          V.generate_random_colors(n=len(popular_ltype),
                                                   seed=5218))
  for gender, marker in zip(['F', 'M'], ['o', '^'])}
legend = {j: i.replace('/', ':') for i, j in styles.items()}

styles_flst = {gender + '/' + laugh: ('red' if 'fl,' in laugh else 'blue', marker)
  for laugh in popular_ltype
  for gender, marker in zip(['F', 'M'], ['o', '^'])}
legend_flst = {s: name[:4] for name, s in styles_flst.items()}
# ====== helper ====== #
def cluster_visualization(feat_name, n_components, perp=30.0,
                          pca_plot=True):
  pca = ds['pca_' + feat_name]
  tsne = TSNE(n_components=2, perplexity=perp,
              early_exaggeration=12.0, learning_rate=120.0,
              n_iter=1200,
              n_iter_without_progress=300,
              min_grad_norm=1e-7,
              metric="euclidean", init="random", verbose=0,
              random_state=5128, method='barnes_hut', angle=0.5)
  # transform data
  laugh_feat = defaultdict(list)
  for lang, laugh, spk, gender, x in iterate_laughter(feat_name):
    x = np.mean(pca.transform(x, n_components=n_components),
                axis=0, keepdims=True)
    name = gender + '/' + laugh
    laugh_feat[name].append(x)
  # merge all samples
  laugh_pca = {i: np.concatenate(j, axis=0)
               for i, j in laugh_feat.items()}
  # post processing
  X = []
  color = []; marker = []
  color_flst = []; marker_flst = []
  for i, j in laugh_pca.items():
    X.append(j)
    n = len(j)
    color += [styles[i][0]] * n
    marker += [styles[i][1]] * n
    color_flst += [styles_flst[i][0]] * n
    marker_flst += [styles_flst[i][1]] * n
  X = np.concatenate(X, axis=0)
  # plot PCA
  if pca_plot:
    plt.figure()
    V.plot_scatter(X[:, 0], X[:, 1], size=4.0,
                   color=color, marker=marker,
                   legend=legend,
                   legend_ncol=len(popular_ltype),
                   fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.suptitle('2D "%s" PCA visualization of laughter and gender' % feat_name)
  # TSNE fitting
  with UnitTimer(name='TSNE fit_transform: %s' % feat_name):
    X = tsne.fit_transform(X)
  # plot TNSE all laugh
  plt.figure()
  V.plot_scatter(X[:, 0], X[:, 1], size=4.0,
                 color=color, marker=marker,
                 legend=legend,
                 legend_ncol=len(popular_ltype),
                 fontsize=10)
  if n_components is None:
    n_components = ds[feat_name].shape[1]
  plt.title('PCA-#components:%d  Perplexity:%d' % (n_components, perp))
  plt.xticks([])
  plt.yticks([])
  plt.suptitle(
      '2D "%s" T-SNE visualization of laughter & gender' % feat_name)
  # plot TNSE flst
  plt.figure()
  V.plot_scatter(X[:, 0], X[:, 1], size=4.0,
                 color=color_flst, marker=marker_flst,
                 legend=legend_flst,
                 legend_ncol=len(popular_ltype),
                 fontsize=10)
  if n_components is None:
    n_components = ds[feat_name].shape[1]
  plt.title('PCA-#components:%d  Perplexity:%d' % (n_components, perp))
  plt.xticks([])
  plt.yticks([])
  plt.suptitle(
      '2D "%s" T-SNE visualization of laughter & gender' % feat_name)
# ====== test ====== #
cluster_visualization('mspec', n_components=10, perp=10, pca_plot=True)
cluster_visualization('mspec', n_components=10, perp=20, pca_plot=False)
cluster_visualization('mspec', n_components=10, perp=30, pca_plot=False)
cluster_visualization('mspec', n_components=20, perp=10, pca_plot=False)
cluster_visualization('mspec', n_components=20, perp=20, pca_plot=False)
cluster_visualization('mspec', n_components=20, perp=30, pca_plot=False)

cluster_visualization('bnf', n_components=10, perp=10, pca_plot=True)
cluster_visualization('bnf', n_components=10, perp=20, pca_plot=False)
cluster_visualization('bnf', n_components=10, perp=30, pca_plot=False)
cluster_visualization('bnf', n_components=20, perp=10, pca_plot=False)
cluster_visualization('bnf', n_components=20, perp=20, pca_plot=False)
cluster_visualization('bnf', n_components=20, perp=30, pca_plot=False)

cluster_visualization('mfcc', n_components=10, perp=10, pca_plot=True)
cluster_visualization('mfcc', n_components=10, perp=20, pca_plot=False)
cluster_visualization('mfcc', n_components=10, perp=30, pca_plot=False)
cluster_visualization('mfcc', n_components=20, perp=10, pca_plot=False)
cluster_visualization('mfcc', n_components=20, perp=20, pca_plot=False)
cluster_visualization('mfcc', n_components=20, perp=30, pca_plot=False)

cluster_visualization('f0', n_components=None, perp=5, pca_plot=True)
cluster_visualization('f0', n_components=None, perp=10, pca_plot=False)
cluster_visualization('f0', n_components=None, perp=20, pca_plot=False)
cluster_visualization('f0', n_components=None, perp=30, pca_plot=False)
cluster_visualization('f0', n_components=None, perp=50, pca_plot=False)

cluster_visualization('loudness', n_components=None, perp=5, pca_plot=True)
cluster_visualization('loudness', n_components=None, perp=10, pca_plot=False)
cluster_visualization('loudness', n_components=None, perp=20, pca_plot=False)
cluster_visualization('loudness', n_components=None, perp=30, pca_plot=False)
cluster_visualization('loudness', n_components=None, perp=50, pca_plot=False)

# ===========================================================================
# Others
# ===========================================================================
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
