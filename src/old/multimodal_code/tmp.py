from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,gpu,tensorflow,cnmem=0.2,seed=1208'
from six.moves import cPickle

import numpy as np
np.random.seed(1208)

from odin import backend as K, nnet as N, fuel as F
from odin.utils import get_modelpath, pad_sequences, pad_center
from utils import evaluate, get_anno, get_data, CODE_PATH

# estmspecF1201lau_mul


def test2():
    name = 'est_texts'
    path = get_modelpath(name, override=False)
    f, args = cPickle.load(open(path, 'r'))
    print(args)
    data_matrix = cPickle.load(
        open(os.path.join(CODE_PATH, 'nlp', '%s_matrix' % args['ds']), 'r'))
    # ====== extract features ====== #
    longest_conversation = data_matrix['longest_conversation'][0]
    X = [i[4] for i in data_matrix['C_01_FF_01_02']]
    tmp = []
    for x in X:
        shape = (longest_conversation, x.shape[1])
        _ = np.zeros(shape=shape)
        _[-x.shape[0]:] = x
        tmp.append(_.reshape((1,) + shape))
    tmp = np.concatenate(tmp, axis=0)
    # ====== make pred ====== #
    X = K.placeholder(shape=(None,) + tmp.shape[1:], name='X')
    K.set_training(False); y = f(X)
    f_pred = K.function(X, y)
    y_pred = f_pred(tmp)
    print(y_pred.ravel().tolist())


def test1():
    name = 'estmfccF481bin_mul'
    path = get_modelpath(name, override=False)
    f, args = cPickle.load(open(path, 'r'))
    print(args)
    train, valid, test = get_data(args['ds'], args['feat'], args['stack'],
                                  args['ctx'], args['hop'], args['mode'],
                                  args['bmode'], ncpu=2)
    print(test.shape)
    X = K.placeholder(shape=(None,) + test.shape[1:], name='X')
    K.set_training(False); y = f(X)
    f_pred = K.function(X, y)

    for name, X, y in test:
        n = 0
        y_pred = []
        while n < X.shape[0]:
            y_pred.append(f_pred(X[n:n + 256]))
            n += 256
        y_pred = np.concatenate(y_pred, axis=0)
        print(name, y_pred.shape, y.shape)
        cPickle.dump((name, y_pred, y),
                     open('/home/trung/data/test', 'w'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
        exit()

test2()
