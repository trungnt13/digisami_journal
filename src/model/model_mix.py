from __future__ import print_function, division, absolute_import

from odin import nnet as N, backend as K
import tensorflow as tf


@N.ModelDescriptor
def mix(X, gender, topic, y, nb_classes):
    f = N.Sequence([
        N.Dimshuffle(pattern=(0, 1, 2, 'x')),

        N.Conv(num_filters=32, filter_size=(7, 7), strides=1,
               b_init=None, pad='same', activation=K.linear),
        N.BatchNorm(axes='auto', activation=K.relu),
        N.Pool(pool_size=(3, 3), strides=2, mode='max'),

        N.Conv(num_filters=64, filter_size=(5, 5), strides=1,
               b_init=None, pad='same', activation=K.linear),
        N.BatchNorm(axes='auto', activation=K.relu),
        N.Pool(pool_size=(3, 3), strides=2, mode='max'),

        N.Conv(num_filters=128, filter_size=(3, 3), strides=1,
               b_init=None, pad='same', activation=K.linear),
        N.BatchNorm(axes='auto', activation=K.relu),
        N.Pool(pool_size=(3, 3), strides=2, mode='max'),

        N.Flatten(outdim=3),
        N.CudnnRNN(num_units=32,
            rnn_mode='gru', num_layers=1,
            input_mode='linear',
            bidirectional=True),

        N.Flatten(outdim=2),
        N.Dense(128, activation=K.relu),
        N.Dropout(level=0.5),
        N.Dense(num_units=nb_classes, activation=K.linear)
    ], debug=True)
    # ====== create output variables ====== #
    y_logit = f(X)
    y_prob = tf.nn.softmax(y_logit)
    return {'prob': y_prob, 'logit': y_logit}


@N.ModelDescriptor
def mixgen(X, gender, topic, y, nb_classes):
    f = N.Sequence([
        N.Dimshuffle(pattern=(0, 1, 2, 'x')),

        N.Conv(num_filters=32, filter_size=(7, 7), strides=1,
               b_init=None, pad='same', activation=K.linear),
        N.BatchNorm(axes='auto', activation=K.relu),
        N.Pool(pool_size=(3, 3), strides=2, mode='max'),

        N.Conv(num_filters=64, filter_size=(5, 5), strides=1,
               b_init=None, pad='same', activation=K.linear),
        N.BatchNorm(axes='auto', activation=K.relu),
        N.Pool(pool_size=(3, 3), strides=2, mode='max'),

        N.Conv(num_filters=128, filter_size=(3, 3), strides=1,
               b_init=None, pad='same', activation=K.linear),
        N.BatchNorm(axes='auto', activation=K.relu),
        N.Pool(pool_size=(3, 3), strides=2, mode='max'),

        N.Flatten(outdim=3),
        N.CudnnRNN(num_units=32,
            rnn_mode='gru', num_layers=1,
            input_mode='linear',
            bidirectional=True),

        N.Flatten(outdim=2),
        N.Dense(128, activation=K.relu),
        N.Dropout(level=0.5),
        N.Dense(num_units=nb_classes, activation=K.linear)
    ], debug=True)
    # ====== create output variables ====== #
    X = tf.concat((X, gender), axis=-1)
    y_logit = f(X)
    y_prob = tf.nn.softmax(y_logit)
    return {'prob': y_prob, 'logit': y_logit}


@N.ModelDescriptor
def mixtpc(X, gender, topic, y, nb_classes):
    f = N.Sequence([
        N.Dimshuffle(pattern=(0, 1, 2, 'x')),

        N.Conv(num_filters=32, filter_size=(7, 7), strides=1,
               b_init=None, pad='same', activation=K.linear),
        N.BatchNorm(axes='auto', activation=K.relu),
        N.Pool(pool_size=(3, 3), strides=2, mode='max'),

        N.Conv(num_filters=64, filter_size=(5, 5), strides=1,
               b_init=None, pad='same', activation=K.linear),
        N.BatchNorm(axes='auto', activation=K.relu),
        N.Pool(pool_size=(3, 3), strides=2, mode='max'),

        N.Conv(num_filters=128, filter_size=(3, 3), strides=1,
               b_init=None, pad='same', activation=K.linear),
        N.BatchNorm(axes='auto', activation=K.relu),
        N.Pool(pool_size=(3, 3), strides=2, mode='max'),

        N.Flatten(outdim=3),
        N.CudnnRNN(num_units=32,
            rnn_mode='gru', num_layers=1,
            input_mode='linear',
            bidirectional=True),

        N.Flatten(outdim=2),
        N.Dense(128, activation=K.relu),
        N.Dropout(level=0.5),
        N.Dense(num_units=nb_classes, activation=K.linear)
    ], debug=True)
    # ====== create output variables ====== #
    X = tf.concat((X, topic), axis=-1)
    y_logit = f(X)
    y_prob = tf.nn.softmax(y_logit)
    return {'prob': y_prob, 'logit': y_logit}


@N.ModelDescriptor
def mixall(X, gender, topic, y, nb_classes):
    f = N.Sequence([
        N.Dimshuffle(pattern=(0, 1, 2, 'x')),

        N.Conv(num_filters=32, filter_size=(7, 7), strides=1,
               b_init=None, pad='same', activation=K.linear),
        N.BatchNorm(axes='auto', activation=K.relu),
        N.Pool(pool_size=(3, 3), strides=2, mode='max'),

        N.Conv(num_filters=64, filter_size=(5, 5), strides=1,
               b_init=None, pad='same', activation=K.linear),
        N.BatchNorm(axes='auto', activation=K.relu),
        N.Pool(pool_size=(3, 3), strides=2, mode='max'),

        N.Conv(num_filters=128, filter_size=(3, 3), strides=1,
               b_init=None, pad='same', activation=K.linear),
        N.BatchNorm(axes='auto', activation=K.relu),
        N.Pool(pool_size=(3, 3), strides=2, mode='max'),

        N.Flatten(outdim=3),
        N.CudnnRNN(num_units=32,
            rnn_mode='gru', num_layers=1,
            input_mode='linear',
            bidirectional=True),

        N.Flatten(outdim=2),
        N.Dense(128, activation=K.relu),
        N.Dropout(level=0.5),
        N.Dense(num_units=nb_classes, activation=K.linear)
    ], debug=True)
    # ====== create output variables ====== #
    X = tf.concat((X, gender, topic), axis=-1)
    y_logit = f(X)
    y_prob = tf.nn.softmax(y_logit)
    return {'prob': y_prob, 'logit': y_logit}
