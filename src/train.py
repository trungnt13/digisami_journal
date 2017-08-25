from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'gpu,float32'
from odin.utils import as_tuple_of_shape, ArgController, get_script_path, stdio, Progbar
args = ArgController(
).add('model', 'name of model'
).add('config', 'name of config in const.py'
).add('-bs', 'batch size', 128
).add('-epoch', 'nb epoch', 8
).parse()

from collections import OrderedDict, defaultdict

import numpy as np
from scipy import stats

from odin import backend as K, nnet as N, training

from processing import get_dataset
from const import outpath
from config import *

if args['config'] not in globals():
    raise ValueError("Cannot find feature configuration with name: '%s' in const.py"
        % args['config'])

# ====== Path management ====== #
MODEL_NAME = args.model + '-' + args.config

LOG_PATH = os.path.join(get_script_path(), 'logs')
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
LOG_PATH = os.path.join(LOG_PATH, MODEL_NAME + '.log')

MODEL_PATH = os.path.join(get_script_path(), 'results')
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
MODEL_PATH = os.path.join(MODEL_PATH, MODEL_NAME)
stdio(path=LOG_PATH)

print("Log path:", LOG_PATH)
print("Model path:", MODEL_PATH)

# ===========================================================================
# Dataset
# ===========================================================================
train, valid, test, nb_classes = get_dataset(**globals()[args.config])
X = [K.placeholder(shape=(None,) + s[1:], dtype='float32', name='input%d' % i)
     for i, s in enumerate(as_tuple_of_shape(train.shape))]
y = K.placeholder(shape=(None, nb_classes), dtype='float32', name='laughter')
inputs = X + [y, nb_classes]
print('Inputs:', inputs)

# ===========================================================================
# Get model
# ===========================================================================
model = N.get_model_descriptor(name=args['model'])
outputs = model(*inputs)

cost, acc, prob = outputs['cost'], outputs['acc'], outputs['prob']
parameters = model.parameters

cm = K.metrics.confusion_matrix(prob, y_true=y, labels=nb_classes)

print(model)
optimizer = K.optimizers.RMSProp(lr=0.00001,
    decay_steps=train.shape[0][0] // args['bs'],
    decay_rate=0.9)
updates = optimizer(outputs['cost'], parameters)

# ====== Functions ====== #
print('Building training functions ...')
f_train = K.function(model.placeholders, [cost, optimizer.norm, cm],
                     updates=updates, training=True)
print('Building testing functions ...')
f_test = K.function(model.placeholders, [cost, outputs['acc'], cm],
                    training=False)
print('Building predicting functions ...')
f_pred = K.function(OrderedDict([(i, j)
                                for i, j in model.placeholders.iteritems()
                                if 'input' in j.name]),
                    outputs['prob'], training=False)

# ===========================================================================
# Build trainer
# ===========================================================================
# print('Start training ...')
# task = training.MainLoop(batch_size=args['bs'], seed=120825, shuffle_level=2,
#                          allow_rollback=True)
# task.set_save(MODEL_PATH, model)
# task.set_callbacks([
#     training.NaNDetector(),
#     training.EarlyStopGeneralizationLoss('valid', cost,
#                                          threshold=5, patience=5)
# ])
# task.set_train_task(f_train, train, epoch=args.epoch, name='train')
# task.set_valid_task(f_test, valid,
#                     freq=training.Timer(percentage=0.4), name='valid')
# task.run()


# ===========================================================================
# Evaluate
# ===========================================================================
def sort_by_idx(pred):
    return np.concatenate([y
                           for i, y in sorted(pred, key=lambda x: x[0])],
            axis=0)


def sort_by_name(segs):
    # sort by start time of the segment name
    X = [x for _, x in sorted(segs,
                              key=lambda x: float(x[0].split(':')[1]))]
    X = np.concatenate(X, axis=0)
    return X


def gender_process(x):
    # get the non-zero mode element
    x = [i.ravel() for i in x]
    x = [i[np.nonzero(i)[0]] for i in x]
    x = np.array([0 if len(i) == 0 else stats.mode(i)[0][0]
                  for i in x])
    return x


def topic_process(x):
    x = stats.mode(x, axis=1)[0]
    return x.ravel()


def report_performance(y_pred, y_true):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    print("===== Frame Prediction =====")
    pred = np.argmax(np.concatenate(y_pred, 0), axis=-1)
    true = np.concatenate(y_true, 0)
    print("Accuracy:", accuracy_score(true, pred))
    print("F1:", f1_score(true, pred, average='micro'))
    print("Confusion matrix:")
    print(confusion_matrix(true, pred))

# ====== init ====== #
y_pred = defaultdict(lambda: defaultdict(list))
y_true = defaultdict(lambda: defaultdict(list))
gender = defaultdict(lambda: defaultdict(list))
topic = defaultdict(lambda: defaultdict(list))
# ====== predict ====== #
prog = Progbar(target=test.shape[0][0])
for name, idx, X, gen, tpc, y in test:
    short_name = name.split(':')[0]
    prog['File'] = short_name
    prog['Name'] = name
    prog['Index'] = idx
    prog.add(X.shape[0])
    y_pred[short_name][name].append((idx, f_pred(X)))
    y_true[short_name][name].append((idx, y))
    gender[short_name][name].append((idx, gen))
    topic[short_name][name].append((idx, tpc))
print()
# ====== post processing ====== #
y_pred = {f: sort_by_name((name, sort_by_idx(pred))
                          for name, pred in segs.iteritems())
          for f, segs in y_pred.iteritems()}
y_true = {f: sort_by_name((name, sort_by_idx(pred))
                          for name, pred in segs.iteritems())
          for f, segs in y_true.iteritems()}
gender = {f: gender_process(sort_by_name((name, sort_by_idx(pred))
                                         for name, pred in segs.iteritems()))
          for f, segs in gender.iteritems()}
topic = {f: topic_process(sort_by_name((name, sort_by_idx(pred))
                                       for name, pred in segs.iteritems()))
         for f, segs in topic.iteritems()}
# ====== check shape matching ====== #
assert set(y_pred.keys()) == set(y_true.keys())
for name in y_pred.keys():
    assert y_pred[name].shape[0] == y_true[name].shape[0] == \
    gender[name].shape[0] == topic[name].shape[0]
exit()
print("*********************************")
print("* Test")
print("*********************************")
report_performance(y_pred, y_true)
