from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
import wave
import shutil
import numpy as np
from math import ceil
from six.moves import cPickle

import numpy as np

from odin import fuel as F, visual as V
from odin.utils import get_all_files
from const import inpath, outpath, CUT_DURATION


# ===========================================================================
# Helper
# ===========================================================================
def array2D(x):
    if x.ndim == 1:
        x = x[None, :]
    return x


def get_partitions(f):
    f = wave.open(f, mode='r')
    duration = int(ceil(f.getnframes() / f.getframerate()))
    f.close()
    partitions = list(range(0, duration - CUT_DURATION // 2, CUT_DURATION)) + [duration]
    partitions = zip(partitions, partitions[1:])
    return partitions

# ===========================================================================
# Getting all segments
# ===========================================================================
segments = []
laugh = {}
topic = {}
gender = {} # lang -> {spkID -> F or M}
for path in inpath:
    lang = 'est' if 'estonian' in path else ('fin' if 'finnish' in path else
                                             'sam')
    print("Input path:", path)
    # ====== read laugh anno ====== #
    laugh.update(
        {lang + '/' + os.path.basename(f).replace('.csv', ''):
            [(spkID, float(start), float(end), text)
         for (spkID, start, end, text) in array2D(np.genfromtxt(f,
            dtype='str', delimiter=':', skip_header=3))]
         for f in get_all_files(os.path.join(path, 'laugh_csv'))}
    )
    # ====== read audio ====== #
    audio_path = os.path.join(path, 'audio')
    segs = [(lang + '/' + os.path.basename(f).replace('.wav', ''), f, 0, -1, 0)
            for f in get_all_files(audio_path, lambda x: '.wav' == x[-4:])]
    # only add segments that contains laugh annotation
    segs = [(n, p, s, e, c) for n, p, s, e, c in segs if n in laugh]
    segments += segs
    # ====== read topic anno ====== #
    topic.update(
        {lang + '/' + os.path.basename(f).replace('.csv', ''):
            [(float(start), float(end), text)
        for (start, end, text) in np.genfromtxt(f,
            dtype='str', delimiter=':', skip_header=3)]
        for f in get_all_files(os.path.join(path, 'topic_csv'))}
    )
    # ====== update speaker gender ====== #
    if lang in ('est', 'fin'):
        gender[lang] = {spkID: spkID[-1]
                        for name, anno in laugh.iteritems()
                        for spkID, _, _, _ in anno
                        if lang + '/' in name}
    else:
        sami_gender = np.genfromtxt(os.path.join(path, 'gender.csv'),
                                    dtype=str, delimiter=' ', skip_header=1)
        gender['sam'] = {spkID: gen for spkID, gen in sami_gender}
print("Audio: %d (files)" % len(segments))
print("Laugh: %d (files)" % len(laugh))
print("Topic: %d (files)" % len(topic))
print("Gender: %d (spk)" % sum(len(i) for i in gender.values()))


# ===========================================================================
# Processing
# ===========================================================================
if os.path.exists(outpath):
    shutil.rmtree(outpath)
# ====== FEAT ====== #
feat = F.SpeechProcessor(segments, outpath,
            sr=None, sr_info={}, sr_new=16000,
            win=0.02, hop=0.01, window='hann',
            nb_melfilters=40, nb_ceps=13,
            get_spec=True, get_qspec=False, get_phase=False,
            get_pitch=True, get_f0=True,
            get_vad=True, get_energy=True, get_delta=2,
            fmin=64, fmax=None,
            pitch_threshold=0.3, pitch_fmax=360, pitch_algo='swipe',
            vad_smooth=3, vad_minlen=0.1,
            cqt_bins=96, preemphasis=0.97,
            center=True, power=2, log=True, backend='odin',
            pca=True, pca_whiten=False,
            audio_ext=None, maxlen=CUT_DURATION,
            save_raw=True, save_stats=True,
            substitute_nan=None, dtype='float16', datatype='memmap',
            ncache=0.08, ncpu=8)
feat.run()
# ====== save ====== #
with open(os.path.join(outpath, 'laugh'), 'w') as f:
    cPickle.dump(laugh, f)
with open(os.path.join(outpath, 'topic'), 'w') as f:
    cPickle.dump(topic, f)
with open(os.path.join(outpath, 'gender'), 'w') as f:
    cPickle.dump(gender, f)

# ===========================================================================
# Sampling and plotting
# ===========================================================================
ds = F.Dataset(outpath, read_only=True)
files = ds['indices'].items()
np.random.shuffle(files)
files = files[:3]
for name, (start, end) in files:
    end = start + 8000
    s, e = ds['indices_raw'][name]
    print(' ', name, end - start, e - s)
    raw = ds['raw'][s:e][:]
    plt.figure()
    # plt.subplot(4, 1, 1); waveplot(raw, sr=8000)
    # plt.title('Raw')
    plt.subplot(4, 1, 2); plt.plot(ds['energy'][start:end, 0].ravel())
    plt.title('Energy')
    plt.subplot(4, 1, 3); plt.plot(ds['pitch'][start:end, 0].ravel())
    plt.title('Pitch')
    plt.subplot(4, 1, 4); plt.plot(ds['f0'][start:end, 0].ravel())
    plt.title('F0')
    plt.tight_layout()
    plt.suptitle(name)
V.plot_save('/tmp/%s.pdf' % os.path.basename(path), clear_all=True,
            dpi=80)
ds.close()
