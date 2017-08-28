from __future__ import print_function, division, absolute_import

from odin import nnet as N, backend as K
import tensorflow as tf


@N.ModelDescriptor
def fnn(X, gender, topic, y, nb_classes):
    f = N.Sequence([
        N.Flatten(outdim=2),
        N.Dense(num_units=1024, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=1024, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=512, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=nb_classes),
    ], debug=True)
    y_logits = f(X)
    y_prob = tf.nn.softmax(y_logits)
    return {'prob': y_prob, 'logit': y_logits}


@N.ModelDescriptor
def fnngen(X, gender, topic, y, nb_classes):
    f = N.Sequence([
        N.Flatten(outdim=2),
        N.Dense(num_units=1024, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=1024, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=512, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=nb_classes),
    ], debug=True)
    X = tf.concat((X, gender), axis=-1)
    y_logits = f(X)
    y_prob = tf.nn.softmax(y_logits)
    return {'prob': y_prob, 'logit': y_logits}


@N.ModelDescriptor
def fnntpc(X, gender, topic, y, nb_classes):
    f = N.Sequence([
        N.Flatten(outdim=2),
        N.Dense(num_units=1024, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=1024, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=512, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=nb_classes),
    ], debug=True)
    X = tf.concat((X, topic), axis=-1)
    y_logits = f(X)
    y_prob = tf.nn.softmax(y_logits)
    return {'prob': y_prob, 'logit': y_logits}


@N.ModelDescriptor
def fnnall(X, gender, topic, y, nb_classes):
    f = N.Sequence([
        N.Flatten(outdim=2),
        N.Dense(num_units=1024, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=1024, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=512, b_init=None),
        N.BatchNorm(activation=tf.nn.relu),
        N.Dense(num_units=nb_classes),
    ], debug=True)
    X = tf.concat((X, gender, topic), axis=-1)
    y_logits = f(X)
    y_prob = tf.nn.softmax(y_logits)
    return {'prob': y_prob, 'logit': y_logits}


@N.ModelDescriptor
def fnn0(X, gender, topic, y, nb_classes):
    f = N.Sequence([
        N.Flatten(outdim=2),
        N.Dense(num_units=1024, b_init=None),
        N.Dense(num_units=1024, b_init=None),
        N.Dense(num_units=512, b_init=None),
        N.Dense(num_units=nb_classes),
    ], debug=True)
    y_logits = f(X)
    y_prob = tf.nn.softmax(y_logits)
    return {'prob': y_prob, 'logit': y_logits}
