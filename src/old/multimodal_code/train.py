from __future__ import print_function, division, absolute_import

from odin.utils import ArgController, stdio, get_logpath, get_modelpath
args = ArgController(
).add('-ds', 'sami, estonia, finnish', 'estonia'
# for training
).add('-bs', 'batch size', 32
).add('-lr', 'learning rate', 0.0001
).add('-epoch', 'number of epoch', 8
# for features
).add('-feat', 'spec, mfcc, mspec', 'mfcc'
).add('-stack', 'whether stack the features or sequencing them', False
).add('-ctx', 'context, how many frames are grouped', 48
).add('-hop', 'together with context, number of shift frame', 1
).add('-mode', 'binary, laugh, all, emotion', 'binary'
).add('-bmode', 'batch mode: all, mul (multiple)', 'mul'
).add('-ncpu', 'number of CPU for feeder', 6
).parse()
# Identical name for model
MODEL_NAME = (args['ds'][:3] + args['feat'] + str(args['stack'])[0] + str(args['ctx']) +
              str(args['hop']) + args['mode'][:3] + '_' + args['bmode'])
# store log
stdio(path=get_logpath(name=MODEL_NAME, override=True))

import os
os.environ['ODIN'] = 'float32,gpu,tensorflow,cnmem=0.2,seed=1208'
from six.moves import cPickle

import numpy as np
np.random.seed(1208)

from odin import backend as K, nnet as N, fuel as F
from odin import training, visual
from odin.utils import Progbar
from odin.basic import has_roles, WEIGHT, PARAMETER

from utils import get_data, laugh_labels, evaluate
# ===========================================================================
# Const
# ===========================================================================
if args['mode'] == 'binary':
    nb_classes = 1
    final_activation = K.sigmoid
    cost_func = K.binary_crossentropy
    score_func1 = K.binary_crossentropy
    score_func2 = K.binary_accuracy
    labels = range(2)
elif args['mode'] == 'laugh':
    nb_classes = 3
    final_activation = K.softmax
    cost_func = lambda x, y: K.bayes_crossentropy(x, y, nb_classes)
    score_func1 = K.categorical_crossentropy
    score_func2 = K.categorical_accuracy
    labels = range(3)
else:
    nb_classes = len(laugh_labels)
    final_activation = K.softmax
    cost_func = lambda x, y: K.bayes_crossentropy(x, y, nb_classes)
    score_func1 = K.categorical_crossentropy
    score_func2 = K.categorical_accuracy
    labels = range(nb_classes)
# ====== get data ====== #
print("MODEL NAME:", MODEL_NAME)
print("#Classes:", nb_classes)
print("Output Activation:", final_activation)
train, valid, test = get_data(args['ds'], args['feat'], args['stack'],
                              args['ctx'], args['hop'], args['mode'],
                              args['bmode'], args['ncpu'])
print('Train shape:', train.shape)
print('Valid shape:', valid.shape)
print('Test shape:', test.shape)

# prog = Progbar(train.shape[0])
# n = 0
# for X, y in train.set_batch(batch_size=32, seed=1208, shuffle_level=2):
#     print(X.shape, y.shape)
#     print(X)
#     print(y)
#     raw_input()
#     n += X.shape[0]
# print(n, train.shape[0])
# ===========================================================================
# Create model
# ===========================================================================
X = K.placeholder(shape=(None,) + train.shape[1:], name='X')
y = K.placeholder(shape=(None,), name='y', dtype='int32')

if not args['stack']:
    f = N.Sequence([
        N.Dimshuffle((0, 1, 2, 'x')),
        N.Conv(32, (3, 3), strides=1, pad='same', activation=K.relu),
        N.Pool(pool_size=2, mode='max'),
        N.BatchNorm(axes='auto'),

        N.Conv(64, (3, 3), strides=1, pad='same', activation=K.relu),
        N.Pool(pool_size=2, mode='max'),
        N.BatchNorm(axes='auto'),

        N.Conv(64, (3, 3), strides=1, pad='same', activation=K.relu),
        N.Pool(pool_size=2, mode='max'),
        N.BatchNorm(axes='auto'),

        N.Flatten(outdim=2),
        N.Dense(nb_classes, activation=final_activation),
    ], debug=True, name=MODEL_NAME)
else:
    f = N.Sequence([
        N.Dense(512, activation=K.relu),
        N.BatchNorm(axes=0),
        N.Dense(256, activation=K.relu),
        N.BatchNorm(axes=0),
        N.Dense(nb_classes, activation=final_activation),
    ], debug=True, name=MODEL_NAME)

K.set_training(1); y_pred_train = f(X)
K.set_training(0); y_pred_eval = f(X)

weights = [w for w in f.parameters if has_roles(w, WEIGHT)]
L1 = K.L1(weights)
L2 = K.L2(weights)

# ====== cost function ====== #
cost_train = K.mean(cost_func(y_pred_train, y))
cost_pred_1 = K.mean(score_func1(y_pred_eval, y))
cost_pred_2 = K.mean(score_func2(y_pred_eval, y))
if args['mode'] == 'binary':
    y_pred_confuse = K.concatenate([1. - y_pred_eval, y_pred_eval], axis=-1)
else:
    y_pred_confuse = y_pred_eval
confusion_matrix = K.confusion_matrix(y_pred_confuse, y, labels=labels)

optimizer = K.optimizers.RMSProp(lr=args['lr'])
updates = optimizer.get_updates(cost_train,
                                [i for i in f.parameters])

print('Building train function ...')
f_train = K.function([X, y], cost_train, updates)
print('Building score function ...')
f_eval = K.function([X, y], [cost_pred_1, cost_pred_2, confusion_matrix])
print('Building pred function ...')
f_pred = K.function(X, y_pred_eval)

# ===========================================================================
# Create traning
# ===========================================================================
print("Preparing main loop ...")
main = training.MainLoop(batch_size=args['bs'], seed=12082518, shuffle_level=2)
main.set_save(
    get_modelpath(name=MODEL_NAME, override=True),
    [f, args]
)
main.set_task(f_train, data=train,
              epoch=args['epoch'], name='Train')
main.set_subtask(f_eval, data=valid,
                 freq=0.5, name='Valid')
main.set_callback([
    training.ProgressMonitor(name='Train', format='Results: {:.4f}'),
    training.ProgressMonitor(name='Valid', format='Results: {:.4f}, {:.4f}',
                             tracking={2: lambda x: sum(x)}),
    training.NaNDetector(name='Train', patience=2, rollback=True),
    training.History(),
    training.EarlyStopGeneralizationLoss(name='Valid', threshold=5, patience=2),
])
main.run()

# ===========================================================================
# Visualization
# ===========================================================================
main['History'].print_batch('Train')
main['History'].print_epoch('Valid')
try:
    print('[Train] Benchmark batch:', main['History'].benchmark('Train', 'batch_end').mean)
    print('[Train] Benchmark epoch:', main['History'].benchmark('Train', 'epoch_end').mean)
    print('[Valid] Benchmark batch:', main['History'].benchmark('Valid', 'batch_end').mean)
    print('[Valid] Benchmark epoch:', main['History'].benchmark('Valid', 'epoch_end').mean)
except:
    pass
# ====== evaluation ====== #
evaluate(get_modelpath(name=MODEL_NAME, override=False))
