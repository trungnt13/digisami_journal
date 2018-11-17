from __future__ import print_function, division, absolute_import

import os
from six.moves import cPickle
import random

import numpy as np

from odin import fuel as F
from odin.utils import get_modelpath, Progbar, bidict
from odin.stats import freqcount

CODE_PATH = '/home/trung/src/digisami'
DATASET_PATH = '/home/trung/data/%s_audio'
ANNO_PATH = '/home/trung/data/%s_anno'
data_order = ['estonia', 'finnish']
SPLIT = [0.6, 0.8]
SEED = 12082518

laugh_labels = bidict({
    u'': 0,
    u'fl, b': 1,
    u'st, e': 2,
    u'st, b': 3,
    u'fl, e': 4,
    u'st, m': 5,
    u'fl, o': 6,
    u'fl, m': 7,
    u'fl, p': 8,
    u'fl, o, p': 9,
    u'fl, d': 10,
    u'st, o': 11,
    u'st, p': 12
})

laugh_labels_binary = bidict({
    u'': 0,
    u'fl, b': 1,
    u'st, e': 1,
    u'st, b': 1,
    u'fl, e': 1,
    u'st, m': 1,
    u'fl, o': 1,
    u'fl, m': 1,
    u'fl, p': 1,
    u'fl, o, p': 1,
    u'fl, d': 1,
    u'st, o': 1,
    u'st, p': 1
})

laugh_shortened = bidict({
    u'': 0,
    u'fl, b': 1,
    u'st, e': 2,
    u'st, b': 3,
    u'fl, e': 4,
    u'st, o': 5,
    u'st, m': 6,
    u'fl, o': 7,
    u'fl, m': 8,
    u'fl, p': 9
})


# ===========================================================================
# Label maker
# ===========================================================================
def flst(x):
    """fl = 1, st = 2"""
    return 0 if len(x) == 0 else (1 if 'fl' in x else 2)


def bin(x):
    return 1 if len(x) > 0 else 0


def allab(x):
    return laugh_labels[x]


def get_anno(lang):
    f = open(ANNO_PATH % lang, 'r')
    anno = cPickle.load(f)
    f.close()
    return anno


def get_data(lang, feat, stack, ctx, hop, mode, batch_mode, ncpu):
    """
    mode: "binary", "laugh", "all", "emotion"
    batch_mode: "all", "mul"
    """
    path = DATASET_PATH % lang
    if mode == 'binary':
        label_parser = bin
    elif mode == 'laugh':
        label_parser = flst
    else:
        label_parser = allab

    ds = F.Dataset(path, read_only=True)
    indices = np.genfromtxt(ds['indices.csv'], dtype=str, delimiter=' ')
    # ====== split indices ====== #
    n = indices.shape[0]
    np.random.seed(SEED); np.random.shuffle(indices)
    train_indices = indices[:int(SPLIT[0] * n)]
    valid_indices = indices[int(SPLIT[0] * n):int(SPLIT[1] * n)]
    test_indices = indices[int(SPLIT[1] * n):]
    print('#Files:', n,
         '  #Train:', train_indices.shape[0],
         '  #Valid:', valid_indices.shape[0],
         '  #Test:', test_indices.shape[0]
    )
    # ====== create feeder ====== #
    maximum_queue_size = 66
    train = F.Feeder(ds[feat], train_indices, ncpu=ncpu,
        buffer_size=3, maximum_queue_size=maximum_queue_size)
    valid = F.Feeder(ds[feat], valid_indices, ncpu=max(1, ncpu // 2),
        buffer_size=1, maximum_queue_size=maximum_queue_size)
    test = F.Feeder(ds[feat], test_indices, ncpu=max(1, ncpu // 2),
        buffer_size=1, maximum_queue_size=maximum_queue_size)
    # create feature transform
    if ctx <= 1:
        feature_transform = None
    elif stack:
        feature_transform = F.recipes.Stacking(left_context=ctx // 2,
                                               right_context=ctx // 2,
                                               shift=hop)
    else:
        feature_transform = F.recipes.Sequencing(frame_length=ctx,
                                                 hop_length=hop,
                                                 end='pad')
    if batch_mode == 'all':
        batch_filter = lambda data: data
    elif batch_mode == 'mul':
        batch_filter = lambda data: None if len(set(data[-1])) <= 1 else data
    else:
        raise ValueError('invalid batch_mode=%s' % batch_mode)
    # ====== create recipes ====== #
    recipes = [
        F.recipes.Normalization(local_normalize=False,
                                mean=ds['%s_mean' % feat],
                                std=ds['%s_std' % feat]),
        feature_transform
    ]
    # ====== set recipes ====== #
    train.set_recipes([F.recipes.TransLoader(ds['laugh.dict'], dtype=int, label_dict=label_parser)] +
                      recipes +
                      [F.recipes.CreateBatch(batch_filter=batch_filter)])
    valid.set_recipes([F.recipes.TransLoader(ds['laugh.dict'], dtype=int, label_dict=label_parser)] +
                      recipes +
                      [F.recipes.CreateBatch()])
    test.set_recipes([F.recipes.TransLoader(ds['laugh.dict'], dtype=str)] +
                     recipes +
                     [F.recipes.CreateFile(return_name=True)])
    return train, valid, test

# ===========================================================================
# For evaluation
# ===========================================================================
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report)


def report_func(y_true, y_pred, nb_classes):
    def report(true, pred):
        labels = np.unique(true).tolist()
        print('Labels:', labels, 'y_true:', np.unique(true), 'y_pred:', np.unique(pred))
        print('Accuracy:', accuracy_score(true, pred))
        print('F1 weighted:', f1_score(true, pred, labels=labels, average='weighted'))
        print('F1 micro:', f1_score(true, pred, labels=labels, average='micro'))
        print('F1 macro:', f1_score(true, pred, labels=labels, average='macro'))
        print('Confusion matrix:')
        print(confusion_matrix(true, pred, labels=labels))
        # print('Report:')
        # print(classification_report(true, pred, labels=labels))

    # TODO there exist the case: for [0, 1, 2], we predict [0, 1]
    # output is 1, but true value is 0, then calibrate is 0 or 1 ?
    def convert_to_flst(true, pred):
        if nb_classes == 2:
            return [flst(j) if i == 1 and len(j) > 0
                    else (random.choice([1, 2]) if i == 1 and len(j) == 0
                          else i)
                    for i, j in zip(pred, true)]
        elif nb_classes == 3:
            return pred
        else:
            return [flst(laugh_labels[i]) for i in pred]

    def convert_to_all(true, pred):
        alllabels = np.unique([allab(i) for i in true if len(i) > 0])
        if nb_classes == 2:
            return [allab(j) if i == 1 and len(j) > 0
                    else (random.choice(alllabels) if i == 1 and len(j) == 0
                          else i)
                    for i, j in zip(pred, true)]
        elif nb_classes == 3:
            return [allab(j) if (i == 1 and 'fl,' in j) or (i == 2 and 'st,' in j)
                    else (random.choice(alllabels) if i > 0 and len(j) == 0
                          else i)
                    for i, j in zip(pred, true)]
        else:
            return pred

    print('SkyPrediction:')
    hist_pred = freqcount(y_pred)
    for i, j in hist_pred.iteritems():
        print(i, ':', j)
    hist_true = freqcount(y_true)
    print('GroundTrue:')
    for i, j in hist_true.iteritems():
        print(i, ':', j)
    # ====== binary ====== #
    print('\n******** Binary problem:')
    report([bin(i) for i in y_true],
           [1 if i > 0 else 0 for i in y_pred])
    # ====== FL-ST ====== #
    print('\n******** FL-ST problem:')
    report([flst(i) for i in y_true],
           convert_to_flst(y_true, y_pred))
    # ====== ALL ====== #
    print('\n******** ALL %d problem:' % len(laugh_labels))
    report([allab(i) for i in y_true],
           convert_to_all(y_true, y_pred))


def evaluate(model_path, threshold=0.5):
    from odin import backend as K

    if not os.path.exists(model_path):
        model_path = get_modelpath(name=model_path, override=False)
    f, args = cPickle.load(open(model_path, 'r'))

    print('======== Configuration ========')
    for i, j in args.iteritems():
        print(i, ':', j)
    print()

    print('======== Loading data ========')
    test = [get_data(path, args['feat'], args['stack'],
                     args['ctx'], args['hop'], args['mode'],
                     args['bmode'], ncpu=2)[-1]
            for path in data_order]
    for i, j in zip(data_order, test):
        print('Test %s:' % i, j.shape)

    print('Building predict function ...')
    K.set_training(False)
    print('Input shape:', (None,) + test[0].shape[1:])
    X = K.placeholder(shape=(None,) + test[0].shape[1:], name='X')
    f_pred = K.function(X, f(X))

    for name, data in zip(data_order, test):
        print('=' * 30, name, '=' * 30) # print header
        y_true = []
        y_pred = []
        nb_classes = 0
        prog = Progbar(target=data.shape[0], title=name)
        for X, y in data.set_batch(batch_size=args['bs'] * 3, seed=None):
            _ = f_pred(X)
            nb_classes = _.shape[-1]
            _ = _ >= threshold if nb_classes == 1 else np.argmax(_, -1)
            y_pred.append(_.astype('int32'))
            y_true.append(y)
            prog.add(X.shape[0])
        # ====== report ====== #
        prog.update(data.shape[0])
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        report_func(y_true, y_pred, 2 if nb_classes == 1 else nb_classes)
        exit()
