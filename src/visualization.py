from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
from odin.utils import Progbar
from odin import visual as V

from processing import set_inspect_mode, get_dataset

set_inspect_mode(True)
data = get_dataset(dsname=['est', 'fin', 'sam'],
    feats=['mspec', 'mfcc', 'energy', 'pitch', 'f0', 'vad'],
    normalize=['mspec', 'mfcc', 'energy'],
    mode='all')
prog = Progbar(data.shape[0])
for name, idx, mspec, mfcc, energy, pitch, f0, vad, gen, tpc, y in data:
    prog.add(mspec.shape[0])
