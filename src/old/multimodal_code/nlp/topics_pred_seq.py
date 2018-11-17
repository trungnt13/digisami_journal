'''
Dictionary size: 1161
Longest transcription: [26, u"and then I don't have time to study, I didn't have time and that's why it took me more than two years"]
Longest topics: [24, u'the other speaker is using polite forms when speaking to the other, so the female speaker asks the male not to use them']
'''

from __future__ import print_function, division, absolute_import
import os
os.environ['ODIN'] = 'float32,gpu,tensorflow'
import re
from itertools import chain

import numpy as np
import cPickle
np.random.seed(1337)

import spacy

from odin import nnet as N, backend as K
from odin.utils import progbar, get_all_files, mpi, pad_sequences, one_hot, Progbar
from odin.fuel import MmapDict, load_glove
from odin.basic import EMBEDDING, has_roles

embedding_ndim = 100
embedding = load_glove(ndim=embedding_ndim)
nlp = spacy.load('en')
invalid_text = re.compile('\*.*\*') # this is expression like *laugh*


# ===========================================================================
# Helper
# ===========================================================================
replacement = [
    ('\n', ''),
    ('Toomas', 'Tomas'),
    (',', ''),
    ("\r", " ")
]


def normalize_sentence(s):
    s = s.strip()
    for i, j in replacement:
        s = s.replace(i, j)
    return s


def extract_tokens(sent):
    for i in invalid_text.findall(sent):
        sent = sent.replace(i, '')
    sent = normalize_sentence(sent)
    tokens = []
    for t in nlp(sent):
        if t.ent_type_ == "": # just token not entity
            t = t.lemma_.lower().strip() if t.lemma_ != "-PRON-" else t.lower_
        elif t.ent_iob_ in ("I", "L"): # inner token of multi-tokens entity
            continue
        else:
            t = t.ent_type_.lower() if t.string.strip() not in embedding \
                else t.lemma_.lower().strip() if t.lemma_ != "-PRON-" else t.lower_
        if t in embedding:
            tokens.append(t)
    return tokens


# ===========================================================================
# Load text
# ===========================================================================
estonia_data = cPickle.load(open('/home/trung/Downloads/all_data', 'r'))
dictionary = []
longest_trans = []
longest_topics = []
longest_conversation = []
X = []
y = []
# ====== create dictionary ====== #
for fname, topics in estonia_data['topics'].iteritems():
    transcriptions = estonia_data['transcriptions'][fname]
    for tp, start, end in topics:
        tp = tp.lower() # for topic it is ok to lower
        tp_tokens = extract_tokens(tp)
        # set longest topics
        if len(tp_tokens) > len(longest_topics):
            longest_topics = tp_tokens
        dictionary += tp_tokens
        conversation = []
        for text, s, e, speaker in transcriptions:
            if s >= start and s <= end: # only care about start time
                # normalize the text
                text_tokens = extract_tokens(text)
                if len(text_tokens) == 0:
                    continue
                conversation.append(text_tokens)
                dictionary += text_tokens
                # longest transcription
                if len(text_tokens) > len(longest_trans):
                    longest_trans = text_tokens
        # longest conversation
        if len(conversation) > len(longest_conversation):
            longest_conversation = conversation
        X.append(conversation)
        y.append(tp_tokens)
# ====== create embedding matrix ====== #
dictionary = list(set(dictionary))
print('Dictionary size:', len(dictionary))
print('Longest transcription:', longest_trans)
print('Longest topics:', longest_topics)
print('Longest conversation:', len(longest_conversation))
embedding_matrix = np.array([embedding[i] for i in dictionary],
                            dtype='float32')
print('Embedding shape:', embedding_matrix.shape)
f_embedding = N.Embedding(input_size=len(dictionary),
                          output_size=embedding_ndim,
                          W_init=embedding_matrix)
# all adj, verb, noun from topics
all_topics_words = list(set([w for w in list(chain(*y))
                            if nlp(w)[0].pos_ in ('ADV', 'ADJ', 'VERB', 'NOUN')]))
all_topics_words = {j: i for i, j in enumerate(all_topics_words)} # word -> idx
print('#words topic:', len(all_topics_words))
# ===========================================================================
# Create training data
# ===========================================================================
length = int(max(len(longest_trans),
                 len(longest_topics)))
X = [pad_sequences(i, maxlen=len(longest_trans), dtype='int32',
                   padding='pre', truncating='pre', value=0,
                   transformer=lambda x: dictionary.index(x))
     for i in X]
# y = [pad_sequences([j], maxlen=len(longest_trans), dtype='int32',
#                    padding='pre', truncating='pre', value=0,
#                    transformer=lambda x: dictionary.index(x))[0]
#     for j in y]
_ = np.zeros(shape=(len(y), len(all_topics_words)))
for i, j in enumerate(y):
    for w in j:
        if w in all_topics_words:
            _[i, all_topics_words[w]] = 1
y = _
shuffle_idx = np.random.permutation(len(X))
X = [X[i] for i in shuffle_idx]
y = y[shuffle_idx]

n_train = int(0.8 * len(X))
X_train = X[:n_train]
X_score = X[n_train:]
y_train = y[:n_train]
y_score = y[n_train:]
# ===========================================================================
# Create model
# ===========================================================================
K.set_training(True)
X_input = K.placeholder(shape=(None, len(longest_trans)), dtype='int32')
y_input = K.placeholder(shape=(1, len(all_topics_words)), dtype='float32')

f_conversation = N.Sequence([
    f_embedding,
    N.AutoRNN(num_units=512, rnn_mode='lstm', num_layers=1,
              input_mode='linear', direction_mode='bidirectional',
              name='encoder')[:, -1, :],
    N.Dimshuffle(pattern=('x', 0, 1)),
    N.AutoRNN(num_units=512, rnn_mode='lstm', num_layers=1,
              input_mode='linear', direction_mode='bidirectional',
              name='decoder')[:, -1, :],
    N.Dense(num_units=512, activation=K.relu),
    N.Dense(num_units=y_score.shape[1], activation=K.softmax)
], debug=True)
y_pred = f_conversation(X_input)
parameters = [p for p in f_conversation.parameters if not has_roles(p, EMBEDDING)]

cost = K.categorical_crossentropy(y_pred, y_input)
optz = K.optimizers.Adadelta()
updates = optz.get_updates(cost, parameters)

fn_train = K.function([X_input, y_input], cost, updates=updates)
fn_pred = K.function(X_input, y_pred)

# ===========================================================================
# HeaderTraining process
# ===========================================================================
for epoch in range(20):
    prog = Progbar(target=len(X_train))
    for x, y in zip(X_train, y_train):
        x = np.asarray(x)
        y = y[None, :]
        c = fn_train(x, y)
        # progress
        prog.title = 'Cost:%.4f' % c
        prog.add(1)
    print()
