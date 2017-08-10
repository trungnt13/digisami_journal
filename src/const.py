from __future__ import print_function, division, absolute_import

import os
from six import string_types

import numpy as np

from odin.utils import struct

# ===========================================================================
# Const
# ===========================================================================
CUT_DURATION = 10

inpath = [
    "/mnt/sdb1/digisami_data/estonian",
    "/mnt/sdb1/digisami_data/finnish",
    "/mnt/sdb1/digisami_data/sami_conv"
]

outpath = "/home/trung/data/sami_feat"

######
mspec_pitch_topic = {
    'features': ['mspec', 'pitch'],
    'normalize': ['mspec'],
    'merge_features': lambda x: np.concatenate(x, axis=-1)
}

mspec_pitch = {
    'features': ['mspec', 'pitch'],
    'normalize': ['mspec'],
    'merge_features': lambda x: np.concatenate(x[:-1], axis=-1)
}

######
mspec_pitch_vad_topic = {
    'features': ['mspec', 'pitch', 'vad'],
    'normalize': ['mspec'],
    'merge_features': lambda x: np.concatenate(x, axis=-1)
}

mspec_pitch_vad = {
    'features': ['mspec', 'pitch', 'vad'],
    'normalize': ['mspec'],
    'merge_features': lambda x: np.concatenate(x[:-1], axis=-1)
}

######
mspec_vad = {
    'features': ['mspec', 'vad'],
    'normalize': ['mspec'],
    'merge_features': lambda x: np.concatenate(x[:-1], axis=-1)
}

mspec_vad_topic = {
    'features': ['mspec', 'vad'],
    'normalize': ['mspec'],
    'merge_features': lambda x: np.concatenate(x, axis=-1)
}

######
mspec = {
    'features': ['mspec'],
    'normalize': ['mspec'],
    'merge_features': lambda x: x[0]
}

mspec_topic = {
    'features': ['mspec'],
    'normalize': ['mspec'],
    'merge_features': lambda x: np.concatenate(x, axis=-1)
}


# ===========================================================================
# Helper methods
# model', 'name of model'
# feat', 'name of feature configuration, specified in const.py'
# ctx', 'length of context'
# hop', 'length of hop'
# -mode', 'laughter detection mode: bin, tri or all', 'bin'
# -seq', 'create features using stacking or sequencing', True
# -utopic', 'unite_topics', True
# -ntopic', 'number of topic for LDA', 6
# -bs', 'batch size', 64
# ===========================================================================
def get_model(cfg):
    if isinstance(cfg, string_types):
        cfg = cfg.split('_')
        name, ds, feat, ctx, hop, code = cfg
        seq, utp, mode, ntp = code
        cfg = struct()
        cfg['model'] = name
        cfg['ds'] = ds
        cfg['feat'] = feat
        cfg['ctx'] = int(ctx)
        cfg['hop'] = int(hop)
        cfg['seq'] = bool(seq)
        cfg['utopic'] = bool(utp)
        cfg['mode'] = 'bin' if mode == '0' else ('tri' if mode == '1' else 'all')
        cfg['ntopic'] = int(ntp)
        return cfg
    elif isinstance(cfg, struct):
        mode = cfg['mode']
        mode = 0 if mode == 'bin' else (1 if mode == 'tri' else 2)
        # convert all boolean value to 1 string
        code = [int(cfg['seq']), int(cfg['utopic']), mode, cfg['ntopic']]
        code = ''.join([str(i) for i in code])
        return '%s_%s_%s_%d_%d_%s' % (cfg['model'], cfg['ds'],
                                      cfg['feat'], cfg['ctx'],
                                      cfg['hop'], code)
    raise ValueError("Config must be string (model name or path), or dict for "
        "arguments configuration.")
