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
import matplotlib.mlab as mlab
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

# ===========================================================================
# Const
# ===========================================================================
ds = F.Dataset(featpath, read_only=True)
with open(os.path.join(ds.path, 'laugh'), 'rb') as f:
    laugh = cPickle.load(f)
with open(os.path.join(ds.path, 'gender'), 'rb') as f:
    gender = cPickle.load(f)
with open(os.path.join(ds.path, 'utt_duration'), 'rb') as f:
    utt_duration = cPickle.load(f)
# ====== get length ====== #
data, nb_classes = get_dataset(dsname=['est', 'fin', 'sam'],
                               feats=['spec', 'mspec', 'mfcc',
                                     'energy', 'pitch', 'sap'],
                               mode='all', ncpu=None,
                               return_single_data=True)
data.set_batch(batch_size=512, batch_mode='file', seed=None)
# ====== fast check the dataset ====== #
if False:
    n = 0
    prog = Progbar(target=len(data), print_report=True, print_summary=True)
    for name, idx, spec, mspec, mfcc, energy, pitch, sap, gen, tpc, y in data:
        assert spec.shape[0] == mspec.shape[0] == mfcc.shape[0] == energy.shape[0] == \
            pitch.shape[0] == sap.shape[0] == gen.shape[0] == \
            tpc.shape[0] == y.shape[0]
        n += mspec.shape[0]
        prog['Name'] = name
        prog['Idx'] = idx
        prog.add(mspec.shape[0])
    assert n == data.shape[0][0]
# ===========================================================================
# Histogram
# ===========================================================================
# language -> Gender -> [laugh events
laugh_duration = defaultdict(lambda: defaultdict(list))
# language -> utterance -> list(fl, st, m, b, e, d, p, o)
laugh_stat = defaultdict(lambda: defaultdict(lambda: [0] * 8))
laugh_order = ['fl, ', 'st, ',
               ', m', ', b', ', e', ', d', ', p', ', o']
for name, events in laugh.items():
    lang = name.split('/')[0]
    utt_name = utt_id(name)
    for start, end, ltype, spk in events:
        gen = gender[lang][spk]
        duration = (end - start) * STEP_LENGTH
        laugh_duration[lang][gen].append((ltype, duration))
        # laugh stats
        for i, j in enumerate(laugh_order):
            if j in ltype:
                laugh_stat[lang][utt_name][i] += 1
# ====== calculate statistic table ====== #
for lang, _ in laugh_stat.items():
    print(lang)
    _ = sorted(_.items(), key=lambda x: x[0])
    all_dur = 0
    all_per = 0
    matx = []
    for utt, stat in _:
        matx.append(stat)
        dur = utt_duration[utt_id(utt)] / 60
        all_dur += dur
        total = stat[0] + stat[1] # fl + st
        per = total / dur
        all_per += per
        dur = timeFMT(dur)
        utt = utt.split('/')[1]
        stat = [str(i) for i in stat]
        print(' & '.join([utt, dur] + stat + [str(total), '%.2f' % per]) + ' \\\\')
    print("Total:", ' & '.join([str(i) for i in np.array(matx).sum(0)]))
    print("Mean Dur:", timeFMT(all_dur / len(_)))
    print("Mean Per:", all_per / len(_))
print("ALL LAUGH:", all_ltype)


def get_hist_data(lang, gen, ltype):
    # ltype is a filter function
    return [i[-1] for i in laugh_duration[lang][gen] if ltype(i[0])]


def show_hist(lang):
    def plot_pdf_hist(ax, y):
        mu, sigma = np.mean(y), np.std(y)
        n, bins, patches = ax.hist(y, bins=20, normed=1, range=(0., 9.))
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        y = mlab.normpdf(bins, mu, sigma)
        ax.plot(bins, y, '--', lw=2)
        return ax
    plt.figure()
    #
    ax = plot_pdf_hist(plt.subplot2grid((2, 4), (0, 0), colspan=2),
                       get_hist_data(lang, 'F', ltype=lambda *args: True))
    ax.set_xlabel('Duration (second)', fontsize=13)
    ax.set_ylabel('Probability density', fontsize=13)
    ax.set_title('Female', fontsize=15)
    #
    ax = plot_pdf_hist(plt.subplot2grid((2, 4), (0, 2), colspan=2),
                       get_hist_data(lang, 'M', ltype=lambda *args: True))
    ax.set_title('Male', fontsize=15)
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
occurences_dist = {}
for lang, dat in laugh_duration.items():
    for gen, events in dat.items():
        _ = defaultdict(float)
        _dist = defaultdict(list)
        for ltype, dur in events:
            _[ltype] += dur
            _dist[ltype].append(dur)
        # events = [e[0] for e in events]
        # events = freqcount(events)
        occurences[lang + '-' + gen] = _
        occurences_dist[lang + '-' + gen] = _dist


def show_occurences_bar(lang):
    male = occurences[lang + '-' + 'M']
    male = [male[lt] if lt in male else 0
            for lt in all_ltype]
    female = occurences[lang + '-' + 'F']
    female = [female[lt] if lt in female else 0
              for lt in all_ltype]
    N = len(all_ltype)
    ind = np.arange(N)    # the x locations for the groups
    width = 0.5       # the width of the bars: can also be len(x) sequence

    plt.figure()
    p1 = plt.bar(ind, male, width)
    p2 = plt.bar(ind, female, width, bottom=male)
    plt.ylabel('Total duration (second)')
    plt.title('')
    plt.xticks(ind, all_ltype, fontsize=13)
    plt.yticks(fontsize=15)
    plt.ylim(ymin=0, ymax=160)
    # plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Men', 'Women'))
    plt.suptitle(alias2name(lang), fontsize=25)

show_occurences_bar(lang='sam')
show_occurences_bar(lang='est')
show_occurences_bar(lang='fin')


def show_occurences_dist_box(lang):
    male = occurences_dist[lang + '-' + 'M']
    male = [male[lt] if lt in male else []
            for lt in all_ltype]
    female = occurences_dist[lang + '-' + 'F']
    female = [female[lt] if lt in female else []
              for lt in all_ltype]
    N = len(all_ltype)
    ind = np.arange(1, N + 1)    # the x locations for the groups

    plt.figure()

    ax = plt.subplot(2, 1, 1)
    plt.boxplot(female, showfliers=False)
    plt.ylabel('Duration (second)', fontsize=15)
    plt.xticks([], [])
    plt.yticks(fontsize=15)
    ax.set_title('Female', fontsize=14)

    ax = plt.subplot(2, 1, 2)
    plt.boxplot(male, showfliers=False)
    plt.xticks(ind, all_ltype, fontsize=13)
    plt.yticks(fontsize=13)
    ax.set_title('Male', fontsize=14)

    # plt.suptitle(alias2name(lang), fontsize=25)

show_occurences_dist_box(lang='sam')
show_occurences_dist_box(lang='est')
show_occurences_dist_box(lang='fin')


# ===========================================================================
# Time Laughter occurences (spectrogram)
# ===========================================================================
def show_spectrogram(resolution, show_title, show_xlab, xtik=None):
    def plot_time_spec(ax, y):
        ax.imshow(y, cmap='Reds')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # language -> {popular_laugh -> time_spec_matrices}
    language_spec = defaultdict(
        lambda: {i: np.zeros(shape=(resolution // 5, resolution))
                 for i in popular_ltype})
    for name in laugh.keys():
        lang = name.split('/')[0]
        n = ds['indices'][name] # second/point
        n = n[1] - n[0]
        events = laugh[name]
        total_laugh = sum(end - start
                          for start, end, ltype, spk in events)
        # create spectrogram for each file
        for start, end, ltype, spk in events:
            if ltype not in popular_ltype:
                continue
            weight = (end - start) / total_laugh
            start = int(np.round(start / n * resolution))
            end = int(np.round(end / n * resolution))
            if end == start:
                end += 1
            language_spec[lang][ltype][:, start:end] += weight

    for lang in ('sam', 'est', 'fin'):
        plt.figure(figsize=(6, 6))
        spec = language_spec[lang]
        for i, ltype in enumerate(popular_ltype):
            y = spec[ltype]
            ax = plt.subplot(6, 1, i + 1)
            plot_time_spec(ax, y)
            ax.set_ylabel(ltype, fontsize=16)
        if xtik is not None:
            plt.xticks(np.arange(len(xtik)), xtik,
                       rotation=60, fontsize=15)
        if show_xlab:
            ax.set_xlabel("Relative timeline", fontsize=15)
        if show_title:
            plt.suptitle(alias2name(lang), fontsize=28)
        plt.subplots_adjust(hspace=0.08)

show_spectrogram(resolution=5, show_title=True, show_xlab=False,
            xtik=['Opening', 'Feedforward', 'Discuss', 'Feedback', 'Closing'])
show_spectrogram(resolution=100, show_title=False, show_xlab=True, xtik=None)

V.plot_save(save_path)
exit()
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
        tpc, gen, lgh) = X[2:]
    # get all laugh events
    for i1, i2, i3, i4 in zip(pitch, f0, gen, lgh):
        if i4 > 0:
            i3 = int(i3); i4 = int(i4)
            laugh_name = str(ALL_LAUGH[i4])
            gender_name = str(GENDER[i3])
            laugh_pitch[lang + '-' + gender_name].append(i1[0])
            laugh_f0[lang + '-' + gender_name].append(i2[0])

# ===========================================================================
# Save everything
# ===========================================================================
V.plot_save(save_path)
