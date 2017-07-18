from __future__ import print_function, division, absolute_import

import os
import shutil

from odin import fuel as F
from odin.utils import get_all_files

inpath = [
    "/mnt/sdb1/digisami_data/estonian/audio",
    "/mnt/sdb1/digisami_data/finnish/audio"
    "/mnt/sdb1/digisami_data/sami_conv/audio"
]

outpath = [
    "/home/trung/data/estonian",
    "/home/trung/data/finnish"
    "/home/trung/data/sami_conv"
]

for opath in outpath:
    if os.path.exists(opath):
        shutil.rmtree(opath)

for ipath, opath in zip(inpath, outpath):
    print("Input path:", ipath)
    print("Output path:", opath)
    segments = [
        (os.path.basename(f).replace('.wav', ''), f, 0, -1, 0)
        for f in get_all_files(ipath, lambda x:'.wav' == x[-4:])]
    print(segments)
    exit()
    feat = F.SpeechProcessor(segments, opath, sr=None, sr_new=8000,
                win=0.02, hop=0.01, window='hann',
                nb_melfilters=None, nb_ceps=None,
                get_spec=True, get_qspec=False, get_phase=False,
                get_pitch=True, get_f0=True,
                get_vad=True, get_energy=True, get_delta=2,
                fmin=64, fmax=None,
                pitch_threshold=0.3, pitch_fmax=260, pitch_algo='swipe',
                vad_smooth=3, vad_minlen=0.1,
                cqt_bins=96, preemphasis=None,
                center=True, power=2, log=True, backend='odin',
                pca=True, pca_whiten=False,
                audio_ext=None, save_stats=True, substitute_nan=None,
                dtype='float16', datatype='memmap',
                ncache=0.12, ncpu=8)
    feat.run()
