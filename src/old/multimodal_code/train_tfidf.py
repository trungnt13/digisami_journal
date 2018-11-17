from __future__ import print_function, division, absolute_import

from odin.utils import ArgController, stdio, get_logpath, get_modelpath
args = ArgController(
).add('-ds', 'sami, estonia, finnish', 'estonia'
# for training
).add('-bs', 'batch size', 8
).add('-lr', 'learning rate', 0.0001
).add('-epoch', 'number of epoch', 8
# for features
).parse()
# Identical name for model
MODEL_NAME = (args['ds'][:3] + '_texts')
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

from utils import get_data, laugh_labels, evaluate, CODE_PATH, SEED

# ===========================================================================
# Load data
# ===========================================================================
print('Model:', MODEL_NAME)
tokenizer = cPickle.load(
    open(os.path.join(CODE_PATH, 'nlp', '%s_tokenizer' % args['ds']), 'r'))
data_matrix = cPickle.load(
    open(os.path.join(CODE_PATH, 'nlp', '%s_matrix' % args['ds']), 'r'))
for i, j in tokenizer.summary.iteritems():
    print(i, ':', j)

# ===========================================================================
# Extract data
# ===========================================================================
X = []
y = []
longest_conversation = data_matrix['longest_conversation'][0]

for f, data in data_matrix.iteritems():
    if f == 'longest_conversation':
        continue
    for (topic, topic_seq, convs_seq,
         topic_tfidf, convs_tfidf,
         laugh, alllaugh, time) in data:
        shape = (longest_conversation, topic_tfidf.shape[1])
        x = np.zeros(shape=shape)
        x[-convs_tfidf.shape[0]:] = convs_tfidf
        # ====== store ====== #
        X.append(x.reshape((1,) + shape))
        y.append(len(alllaugh))
# ====== finalize data ====== #
X = np.concatenate(X, axis=0)
y = np.array(y, dtype='float32')
y = (y - np.min(y)) / (np.max(y) - np.min(y))
print('Data Shape:', X.shape, y.shape)
# ====== train test split ====== #
np.random.seed(SEED)
n = len(y)
idx = np.random.permutation(n)
X = X[idx]; y = y[idx]
SPLIT = 0.8
X_train = X[:int(SPLIT * n)]
y_train = y[:int(SPLIT * n)]
X_valid = X[int(SPLIT * n):]
y_valid = y[int(SPLIT * n):]
print('Training:', X_train.shape)
print('Validing:', X_valid.shape)


# ===========================================================================
# Different model
# ===========================================================================
def model1():
    f = N.Sequence([
        N.Dimshuffle(pattern=(0, 1, 'x', 2)),
        N.Conv(num_filters=512, filter_size=(5, 1), pad='valid',
               strides=(1, 1), activation=K.linear),
        N.BatchNorm(activation=K.relu),

        N.Conv(num_filters=256, filter_size=(5, 1), pad='valid',
               strides=(1, 1), activation=K.linear),
        N.BatchNorm(activation=K.relu),

        N.Flatten(outdim=3),
        N.CudnnRNN(num_units=128, rnn_mode='lstm', input_mode='linear',
                   num_layers=2,
                   direction_mode='unidirectional'),
        N.BatchNorm(axes='auto'),

        N.Flatten(outdim=2),
        N.Dense(1, activation=K.sigmoid),
    ], debug=True, name=MODEL_NAME)
    return f


def model3():
    f = N.Sequence([
        N.Dimshuffle(pattern=(0, 1, 2, 'x')),
        N.Conv(num_filters=32, filter_size=(5, 126), pad='valid',
               strides=(1, 1), activation=K.linear),
        N.BatchNorm(activation=K.relu),
        N.Pool(pool_size=(2, 5), mode='max'),

        N.Flatten(outdim=3),
        N.CudnnRNN(num_units=256, rnn_mode='lstm', input_mode='linear',
                   num_layers=2,
                   direction_mode='unidirectional'),
        N.BatchNorm(axes='auto'),

        N.Flatten(outdim=2),
        N.Dense(1, activation=K.sigmoid),
    ], debug=True, name=MODEL_NAME)
    return f


def model2():
    f = N.Sequence([
        N.CudnnRNN(num_units=256, rnn_mode='lstm', input_mode='linear',
                   num_layers=2,
                   direction_mode='bidirectional'),
        N.BatchNorm(axes=0),

        N.Flatten(outdim=2),
        N.Dense(1, activation=K.sigmoid),
    ], debug=True, name=MODEL_NAME)
    return f

# ===========================================================================
# Create model
# ===========================================================================
X_ = K.placeholder(shape=(None,) + X_train.shape[1:], name='X')
y_ = K.placeholder(shape=(None,), name='y', dtype='float32')

f = model2()
K.set_training(1); y_pred_train = f(X_)
K.set_training(0); y_pred_eval = f(X_)

# ====== weights and params ====== #
weights = [w for w in f.parameters if has_roles(w, WEIGHT)]
L1 = K.L1(weights)
L2 = K.L2(weights)

params = f.parameters
print('Params:', [p.name for p in params])
# ====== cost function ====== #
cost_train = K.mean(K.binary_crossentropy(y_pred_train, y_))
cost_pred_1 = K.mean(K.binary_crossentropy(y_pred_eval, y_))
cost_pred_2 = K.mean(K.squared_error(y_pred_eval, y_))

optimizer = K.optimizers.RMSProp(lr=args['lr'])
updates = optimizer.get_updates(cost_train, params)

print('Building train function ...')
f_train = K.function([X_, y_], cost_train, updates)
print('Building score function ...')
f_eval = K.function([X_, y_], [cost_pred_1, cost_pred_2])
print('Building pred function ...')
f_pred = K.function(X_, y_pred_eval)

# ===========================================================================
# Create traning
# ===========================================================================
print("Preparing main loop ...")
main = training.MainLoop(batch_size=args['bs'], seed=12082518, shuffle_level=2)
main.set_save(
    get_modelpath(name=MODEL_NAME, override=True),
    [f, args]
)
main.set_task(f_train, data=(X_train, y_train),
              epoch=args['epoch'], name='Train')
main.set_subtask(f_eval, data=(X_valid, y_valid),
                 freq=0.6, name='Valid')
main.set_callback([
    training.ProgressMonitor(name='Train', format='Results: {:.4f}'),
    training.ProgressMonitor(name='Valid', format='Results: {:.4f}, {:.4f}'),
    # training.NaNDetector(name='Train', patience=3, rollback=True),
    training.History(),
    training.EarlyStopGeneralizationLoss(name='Valid', threshold=5, patience=5),
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
# ===========================================================================
# Evaluate
# ===========================================================================
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def report(ytrue, ypred):
    print()
    print("Accuracy:", accuracy_score(ytrue, ypred))
    print("F1:", f1_score(ytrue, ypred))
    print("Confustion:")
    print(confusion_matrix(ytrue, ypred))
f = cPickle.load(open(get_modelpath(name=MODEL_NAME, override=False), 'r'))[0]
y_pred = f_pred(X_valid).ravel()
y_true = y_valid
for i, j in zip(y_pred, y_true):
    print(i, j)

report(y_true >= 0.1, y_pred >= 0.1)
report(y_true >= 0.5, y_pred >= 0.5)
