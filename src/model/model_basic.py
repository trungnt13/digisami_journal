from __future__ import print_function, division, absolute_import

from odin import nnet as N, backend as K
import tensorflow as tf


@N.ModelDescriptor
def basic1(X, vad, y, nb_classes):
    f = N.Sequence([
        N.Flatten(outdim=2),
        N.Dense(num_units=1024, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=1024, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=512, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=nb_classes),
    ])
    y_logits = f(X)
    y_prob = tf.nn.softmax(y_logits)
    ce = tf.losses.softmax_cross_entropy(y, y_logits)
    acc = K.metrics.categorical_accuracy(y_prob, y)
    return {'prob': y_prob, 'logit': y_logits,
            'cost': ce, 'acc': acc}
