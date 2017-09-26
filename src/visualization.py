# ===========================================================================
# Plot the laugh duration hitogram compared among 3 datasets
# Layout:
# | GeneralLaughSami | Estonian | Finnish | (histogram)
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
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt
import os
from six.moves import cPickle
from collections import defaultdict

import numpy as np
from odin.utils import Progbar
from odin import visual as V
from odin import fuel as F
from odin.stats import freqcount

from const import outpath
from processing import (set_inspect_mode, get_dataset,
                        GENDER, ALL_LAUGH)


# ===========================================================================
# Helper
# ===========================================================================
def alias2name(lang):
    return 'Sami' if lang == 'sam' else \
        ('Finnish' if lang == 'fin' else 'Estonian')

# ===========================================================================
# Const
# ===========================================================================
set_inspect_mode(True)

ds = F.Dataset(outpath, read_only=True)
with open(os.path.join(ds.path, 'laugh')) as f:
    laugh = cPickle.load(f)
with open(os.path.join(ds.path, 'gender'), 'r') as f:
    gender = cPickle.load(f)
# ====== get length ====== #
length = defaultdict(lambda: 0)
for name, (n1, n2) in ds['indices'].iteritems():
    name, start, end = name.split(':')
    # first value is length in second
    length[name] = max(length[name], float(end))
data = get_dataset(dsname=['est', 'fin', 'sam'],
    feats=['spec', 'mspec', 'mfcc', 'energy', 'pitch', 'f0', 'vad'],
    normalize=['spec', 'mspec', 'mfcc', 'energy'],
    mode='all', ncpu=6)
# ====== fast check the dataset ====== #
if False:
    n = 0
    for name, idx, spec, mspec, mfcc, energy, pitch, f0, vad, gen, tpc, y in data:
        assert spec.shape[0] == mspec.shape[0] == mfcc.shape[0] == energy.shape[0] == \
            pitch.shape[0] == f0.shape[0] == vad.shape[0] == gen.shape[0] == \
            tpc.shape[0] == y.shape[0]
        n += mspec.shape[0]
    assert n == data.shape[0][0]
# ===========================================================================
# Histogram
# ===========================================================================
# language -> Gender -> [laugh events
x = defaultdict(lambda: defaultdict(list))
all_ltype = []
for name, events in laugh.iteritems():
    lang = name.split('/')[0]
    all_ltype += [e[-1] for e in events]
    for spk, start, end, ltype in events:
        gen = gender[lang][spk]
        x[lang][gen].append((ltype, end - start))
all_ltype = sorted(set(all_ltype))
all_ltype.remove('fl, o, p')
all_ltype.remove('joo')
all_ltype.remove('fl, d')
print("ALL LAUGH:", all_ltype)


def get_hist_data(lang, gen, ltype):
    # ltype is a filter function
    return [i[-1] for i in x[lang][gen] if ltype(i[0])]


def show_hist(lang):
    def plot_pdf_hist(ax, y):
        mu, sigma = np.mean(y), np.std(y)
        n, bins, patches = ax.hist(y, bins=20, normed=1, range=(0., 9.))
        y = mlab.normpdf(bins, mu, sigma)
        ax.plot(bins, y, '--')
        return ax

    plt.figure()
    #
    ax = plot_pdf_hist(plt.subplot2grid((2, 4), (0, 0), colspan=2),
                       get_hist_data(lang, 'F', ltype=lambda *args: True))
    ax.set_xlabel('Duration (second)')
    ax.set_ylabel('Probability density')
    ax.set_title('Female')
    #
    ax = plot_pdf_hist(plt.subplot2grid((2, 4), (0, 2), colspan=2),
                       get_hist_data(lang, 'M', ltype=lambda *args: True))
    ax.set_title('Male')
    #
    ax = plot_pdf_hist(plt.subplot2grid((2, 4), (1, 0), colspan=1),
                       get_hist_data(lang, 'F', ltype=lambda x: 'st,' in x))
    ax.set_title('st')
    ax = plot_pdf_hist(plt.subplot2grid((2, 4), (1, 1), colspan=1),
                       get_hist_data(lang, 'F', ltype=lambda x: 'fl' in x))
    ax.set_title('fl')
    #
    ax = plot_pdf_hist(plt.subplot2grid((2, 4), (1, 2), colspan=1),
                       get_hist_data(lang, 'M', ltype=lambda x: 'st,' in x))
    ax.set_title('st')
    ax = plot_pdf_hist(plt.subplot2grid((2, 4), (1, 3), colspan=1),
                       get_hist_data(lang, 'M', ltype=lambda x: 'fl' in x))
    ax.set_title('fl')
    #
    plt.suptitle(alias2name(lang), fontsize=18)
    plt.tight_layout()
show_hist(lang='sam')
show_hist(lang='est')
show_hist(lang='fin')

# ===========================================================================
# Gender laughter occurences
# ===========================================================================
occurences = {}
for ds, dat in x.iteritems():
    for gen, events in dat.iteritems():
        events = [e[0] for e in events]
        events = freqcount(events)
        occurences[ds + '-' + gen] = events


def show_occurences_bar(lang):
    male = occurences[lang + '-' + 'M']
    male = [male[lt] if lt in male else 0
            for lt in all_ltype]
    female = occurences[lang + '-' + 'F']
    female = [female[lt] if lt in female else 0
              for lt in all_ltype]
    N = len(all_ltype)
    ind = np.arange(N)    # the x locations for the groups
    width = 0.4       # the width of the bars: can also be len(x) sequence

    plt.figure()
    p1 = plt.bar(ind, male, width)
    p2 = plt.bar(ind, female, width, bottom=male)
    plt.ylabel('Number of occurences')
    plt.title('')
    plt.xticks(ind, all_ltype)
    # plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Men', 'Women'))
    plt.suptitle(alias2name(lang), fontsize=18)

show_occurences_bar(lang='sam')
show_occurences_bar(lang='est')
show_occurences_bar(lang='fin')

# ===========================================================================
# F0 and pitch
# ===========================================================================
print(data)
prog = Progbar(target=data.shape[0][0],
               print_report=True)
laugh_f0 = defaultdict(list)
laugh_pitch = defaultdict(list)

for X in data:
    name = X[0]
    lang, file_name = name.split('/')
    idx = X[1]
    (spec, mspec, mfcc, energy, pitch, f0, vad,
        gen, tpc, y) = X[2:]
    prog.add(spec.shape[0])
    # get all laugh events
    for i1, i2, i3, i4 in zip(pitch, f0, gen, y):
        if i4 > 0:
            i3 = int(i3); i4 = int(i4)
            laugh_name = ALL_LAUGH[i4]
            gender_name = GENDER[i3]
            laugh_pitch[lang + '-' + gender_name].append(i1)
            laugh_f0[lang + '-' + gender_name].append(i2)
laugh_f0 = {name: (np.mean(dat), np.std(dat))
            for name, dat in laugh_f0.iteritems()}
laugh_pitch = {name: (np.mean(dat), np.std(dat))
               for name, dat in laugh_pitch.iteritems()}
print(laugh_f0)
exit()

# ===========================================================================
# Time Laughter occurences (spectrogram)
# ===========================================================================
resolution = 300
# language -> [ST, FL]
x = defaultdict(lambda: [
    np.zeros(shape=(resolution // 5, resolution)),
    np.zeros(shape=(resolution // 5, resolution))
])
for name in laugh.keys():
    lang = name.split('/')[0]
    n = length[name] / resolution # second/point
    events = laugh[name]
    # create spectrogram for each file
    st = np.zeros(shape=(resolution // 5, resolution))
    fl = np.zeros(shape=(resolution // 5, resolution))
    for spk, start, end, ltype in events:
        start = int(start / n)
        end = int(end / n)
        if 'st,' in ltype:
            st[:, start:end] += 1
        elif 'fl,' in ltype:
            fl[:, start:end] += 1
    # normalize
    n = st[0].sum() + fl[0].sum()
    if n > 0:
        st = st / n
        fl = fl / n
        x[lang][0] += st
        x[lang][1] += fl


def show_spectrogram(ax, y):
    ax.imshow(y, cmap='Reds')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_yticklabels([]); ax.set_xticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

plt.figure(figsize=(8, 2))
ax = plt.subplot(2, 3, 1)
show_spectrogram(ax, x['sam'][0])
ax.set_title('Sami')
ax.set_ylabel('st')
ax.set_xlabel('Time')
ax = plt.subplot(2, 3, 4)
show_spectrogram(ax, x['sam'][1])
ax.set_ylabel('fl')
ax.set_xlabel('Time')

ax = plt.subplot(2, 3, 2)
show_spectrogram(ax, x['est'][0])
ax.set_title('Estonian')
ax = plt.subplot(2, 3, 5)
show_spectrogram(ax, x['est'][1])

ax = plt.subplot(2, 3, 3)
show_spectrogram(ax, x['fin'][0])
ax.set_title('Finnish')
ax = plt.subplot(2, 3, 6)
show_spectrogram(ax, x['fin'][1])

plt.tight_layout()
# ===========================================================================
# Save everything
# ===========================================================================
path = '/Users/trungnt13/tmp/tmp.pdf'
if not os.path.exists(path):
    path = '/tmp/tmp.pdf'
V.plot_save(path)
