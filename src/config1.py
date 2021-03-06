from __future__ import print_function, division, absolute_import

longcontext = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mspec'],
    'normalize': ['mspec'],
    'mode': 'bin',
    'context': 80,
    'hop': 10,
    'seq': True,
}

# ===========================================================================
# MFCC
# ===========================================================================
efstri0 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc'],
    'normalize': ['mfcc'],
    'mode': 'tri',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efstri1 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'pitch'],
    'normalize': ['mfcc', 'pitch'],
    'mode': 'tri',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efstri2 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'f0'],
    'normalize': ['mfcc', 'f0'],
    'mode': 'tri',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efstri3 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'energy'],
    'normalize': ['mfcc', 'energy'],
    'mode': 'tri',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efstri4 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'vad'],
    'normalize': ['mfcc'],
    'mode': 'tri',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efstri5 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'energy', 'vad'],
    'normalize': ['mfcc', 'energy'],
    'mode': 'tri',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efstri6 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'pitch', 'energy', 'vad'],
    'normalize': ['mfcc', 'pitch', 'energy'],
    'mode': 'tri',
    'context': 30,
    'hop': 10,
    'seq': True,
}

efstri7 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'pitch', 'f0', 'energy', 'vad'],
    'normalize': ['mfcc', 'pitch', 'f0', 'energy'],
    'mode': 'tri',
    'context': 30,
    'hop': 10,
    'seq': True,
}
