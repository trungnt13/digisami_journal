from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'cpu,float32,cnmem=0.2'

import numpy as np

from odin import backend as K, nnet as N, training

from processing import get_dataset

# ===========================================================================
# Dataset
# ===========================================================================
train, valid, test, nb_classes = get_dataset(
    dsname='estonian',
    features=['mspec', 'pitch', 'vad'],
    normalize=['mspec'],
    gender=False,
    merge_features=lambda x: (np.concatenate(x[:2] + [x[-1]], axis=-1), x[-2]),
    mode='bin',
    context=20, hop=10, seq=False,
    unite_topics=True,
    nb_topics=6,
    ncpu=4, seed=12082518)
X = [K.placeholder(shape=(None,) + s[1:], dtype='float32', name='input%d' % i)
     for i, s in enumerate(train.shape)]
y = K.placeholder(shape=(None, nb_classes), dtype='float32', name='laughter')
inputs = X + [y, nb_classes]
# ===========================================================================
# Get model
# ===========================================================================
model = N.get_model_descriptor(name='basic1')
outputs = model(*inputs)
cost = outputs['cost']
acc = outputs['acc']
prob = outputs['prob']
parameters = model.parameters

cm = K.metrics.confusion_matrix(prob, y_true=y, labels=nb_classes)

print("Inputs:", model.placeholders)
print("#Params:", model.nb_parameters, model.nb_parameters * 4. / 1024 / 1024, "MB")
optimizer = K.optimizers.RMSProp(lr=0.00001,
    decay_steps=None,
    decay_rate=0.96)
updates = optimizer(outputs['cost'], parameters)

# ====== Functions ====== #
print('Building training functions ...')
f_train = K.function(model.placeholders, [cost, optimizer.norm, cm],
                     updates=updates, training=True)
print('Building testing functions ...')
f_test = K.function(model.placeholders, [cost, outputs['acc'], cm],
                    training=False)
print('Building predicting functions ...')
f_pred = K.function(model.placeholders, outputs['prob'], training=False)

# ===========================================================================
# Build trainer
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=128, seed=12, shuffle_level=2,
                         allow_rollback=True, print_progress=True,
                         confirm_exit=True)
task.set_save('/tmp/tmpmodel', model)
task.set_callbacks([
    training.NaNDetector(),
    training.EarlyStopGeneralizationLoss('valid', cost, threshold=5, patience=5)
])
task.set_train_task(f_train, train, epoch=8, name='train')
task.set_valid_task(f_test, valid,
                    freq=training.Timer(percentage=0.4), name='valid')
task.run()

# ===========================================================================
# Evaluate
# ===========================================================================
