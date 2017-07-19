from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
import shutil
import numpy as np
from six.moves import cPickle

from odin import fuel as F, visual as V
from odin.utils import get_all_files
from const import inpath, outpath


# ===========================================================================
# Const
# ===========================================================================
def array2D(x):
    if x.ndim == 1:
        x = x[None, :]
    return x

# ===========================================================================
# Processing
# ===========================================================================
if True:
    for ipath, opath in zip(inpath, outpath):
        print("Input path:", ipath)
        print("Output path:", opath)
        if os.path.exists(opath):
            shutil.rmtree(opath)

        audio_path = os.path.join(ipath, 'audio')
        segments = [
            (os.path.basename(f).replace('.wav', ''), f, 0, -1, 0)
            for f in get_all_files(audio_path, lambda x:'.wav' == x[-4:])]
        print("Found: %d (segments)" % len(segments))
        print(os.path.join(ipath, 'laugh_csv'))
        # ====== read laugh anno ====== #
        laugh = {os.path.basename(f).replace('.csv', ''):
                [(spkID, float(start), float(end), text)
                 for (spkID, start, end, text) in array2D(np.genfromtxt(f,
                    dtype='str', delimiter=':', skip_header=3))]
                 for f in get_all_files(os.path.join(ipath, 'laugh_csv'))}
        # ====== read topic anno ====== #
        topic = {os.path.basename(f).replace('.csv', ''):
                 [(float(start), float(end), text)
                  for (start, end, text) in np.genfromtxt(f,
                    dtype='str', delimiter=':', skip_header=3)]
                 for f in get_all_files(os.path.join(ipath, 'topic_csv'))}
        # ====== FEAT ====== #
        feat = F.SpeechProcessor(segments, opath, sr=None, sr_new=8000,
                    win=0.02, hop=0.005, window='hann',
                    nb_melfilters=40, nb_ceps=13,
                    get_spec=True, get_qspec=False, get_phase=False,
                    get_pitch=True, get_f0=True,
                    get_vad=True, get_energy=True, get_delta=2,
                    fmin=64, fmax=None,
                    pitch_threshold=0.3, pitch_fmax=360, pitch_algo='swipe',
                    vad_smooth=3, vad_minlen=0.1,
                    cqt_bins=96, preemphasis=None,
                    center=True, power=2, log=True, backend='odin',
                    pca=True, pca_whiten=False,
                    audio_ext=None, save_raw=True,
                    save_stats=True, substitute_nan=None,
                    dtype='float16', datatype='memmap',
                    ncache=0.08, ncpu=8)
        feat.run()
        # ====== save ====== #
        with open(os.path.join(opath, 'laugh'), 'w') as f:
            cPickle.dump(laugh, f)
        with open(os.path.join(opath, 'topic'), 'w') as f:
            cPickle.dump(topic, f)

# ===========================================================================
# Sampling and plotting
# ===========================================================================
for path in outpath:
    from librosa.display import waveplot
    print(path)
    ds = F.Dataset(path, read_only=True)
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
