from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'gpu,float32'
from odin.utils import as_tuple_of_shape, ArgController, get_script_path, stdio
args = ArgController(
).add('model', 'name of model'
).add('config', 'name of config in const.py'
).add('-bs', 'batch size', 128
).add('-epoch', 'nb epoch', 8
).parse()

from collections import OrderedDict

import numpy as np

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
    decay_steps=train.shape[0] // args['bs'],
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
print('Start training ...')
task = training.MainLoop(batch_size=args['bs'], seed=120825, shuffle_level=2,
                         allow_rollback=True)
task.set_save(MODEL_PATH, model)
task.set_callbacks([
    training.NaNDetector(),
    training.EarlyStopGeneralizationLoss('valid', cost,
                                         threshold=5, patience=5)
])
task.set_train_task(f_train, train, epoch=args.epoch, name='train')
task.set_valid_task(f_test, valid,
                    freq=training.Timer(percentage=0.4), name='valid')
task.run()


# ===========================================================================
# Evaluate
# ===========================================================================
def report_performance(y_pred, y_true):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    print("===== Frame Prediction =====")
    pred = np.argmax(np.concatenate(y_pred, 0), axis=-1)
    true = np.concatenate(y_true, 0)
    print("Accuracy:", accuracy_score(true, pred))
    print("F1:", f1_score(true, pred, average='micro'))
    print("Confusion matrix:")
    print(confusion_matrix(true, pred))

y_pred = []
y_true = []
for inputs in test:
    print("Predicting:", inputs[0])
    y_pred.append(f_pred(inputs[1:-1]))
    y_true.append(np.argmax(inputs[-1], axis=-1))

print("*********************************")
print("* Test")
print("*********************************")
report_performance(y_pred, y_true)
