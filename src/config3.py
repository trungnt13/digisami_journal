from __future__ import print_function, division, absolute_import

stack0 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc'],
    'normalize': ['mfcc'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': False,
}

stack1 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'pitch'],
    'normalize': ['mfcc', 'pitch'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': False,
}

stack2 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'f0'],
    'normalize': ['mfcc', 'f0'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': False,
}

stack3 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'energy'],
    'normalize': ['mfcc', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': False,
}

stack4 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'vad'],
    'normalize': ['mfcc'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': False,
}

stack5 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'energy', 'vad'],
    'normalize': ['mfcc', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': False,
}

stack6 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'pitch', 'energy', 'vad'],
    'normalize': ['mfcc', 'pitch', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': False,
}

stack7 = {
    'dsname': ['est', 'fin', 'sam'],
    'feats': ['mfcc', 'pitch', 'f0', 'energy', 'vad'],
    'normalize': ['mfcc', 'pitch', 'f0', 'energy'],
    'mode': 'bin',
    'context': 30,
    'hop': 10,
    'seq': False,
}
