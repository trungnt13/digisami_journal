from __future__ import print_function, division, absolute_import

from odin.utils import ArgController, progbar

args = ArgController(
).add('-ds', 'estonia, finnish, sami', 'estonia'
).parse()

import os
import re
from itertools import chain
from six.moves import cPickle
from collections import OrderedDict

from utils import get_anno, CODE_PATH
from odin.preprocessing import text
from odin import fuel as F

# ===========================================================================
# Load data
# ===========================================================================
data = get_anno(args['ds'])

trans = data['transcriptions']
topics = data['topics']
laugh = data['laughter']

files = sorted(trans.keys())
print('Found:', len(files), 'files')
if len(trans.keys()) != len(topics.keys()) or len(trans.keys()) != len(laugh.keys()):
    raise ValueError('length of trans=%, topics=%d, laughter=%d' %
                     (len(trans), len(topics), len(laugh)))


# ===========================================================================
# helper
# ===========================================================================
laugh_regex = re.compile('\*\w*\s?laugh\*')


def filter_trans(t):
    t = t.replace('[...]', '')
    t = t.replace('yea', 'yeah')
    t = t.replace('yeahh', 'yeah')
    t = t.replace('umm', '')
    t = t.replace('um', '')
    t = t.replace('mhm', '')
    t = t.replace('y-you', 'you')
    t = t.replace('yeahrs', 'years')
    t = t.replace('yeahr', 'year')
    t = t.replace('whare', 'where')
    # ====== misspell ====== #
    misspell = {
        "voluntee": "volunteer",
        'usally': 'usually',
        "Tueasday": "Tuesday",
        "Sudnday": "Sunday",
        "stuying": "studying",
        "somewehre": "somewhere",
        "sgain": "again",

        "Unfortunenately": "Unfortunately",
        "Fortunenately": "Fortunately",

        "scolarship": "scholarship",
        "scholarshop": "scholarship",

        "prese": "press",
        "polotics": "politics",
        "oportunity": "opportunity",
        "morphsynthesis": "morphology synthesis",
        "morphanalysis": "morphology analysis",
        "managaged": "managed",
        "kindegartens": "kindergartens",
        "intitute": "institute",
        "inmathematics": "in mathematics",
        "hostpitals": "hospitals",
        "horisontal": "horizontal",
        "gymnasi": "gymnastic",
        "gocery": "grocery",
        "frameanalysis": "frame analysis",
        "fortu": "fortune",
        "excercised": "exercised",

        "aaand": "and",
        "pressnt": "present",
        "webmedia": "web media",

        "exatcly": "exactly",
        "Exacty": "exactly",
        "exactl": "exact",

        "okei": "ok",

        "dialoogid": "dialog ID",
        "dather": "father",
        "culatively": "cumulatively",
        "competetions": "competitions",
        "cinamon": "cinnamon",
        "buut": "but",
        "bussinessmen": "business men",
        "binded": "bind",
        "bevause": "because",
        "bacherlor": "bachelor",
        "bacause": "because",
        "algthough": "although",

        "questionns": "questions",
        "questio": "question",

        "interesing": "interesting",
        "certanly": "certainly",
        "biotechincal": "bio-technical",
        "youh": "youth",
        "valueable": "valuable",

        "yees": "yes",
        "yeeaa": "yeah",
        "yeahah": "yeah",
        "yeyeah": "yeah",
        "yesyes": "yes",
        "yeahyeah": "yeah",
        "yeahm": "yeah",
        "yeahaah": "yeah",
        "yeahaaah": "yeah",
        "yeaha": "yeah",

        "acutally": "actually",
        "actualy": "actually",
        "actally": "actually",
        "acually": "actually",

        "alreayd": "already",
        "basicly": "basically",
        "bussiness": "business",
        "*gasp*": "",
        "*tsk*": ""
    }
    for i, j in misspell.iteritems():
        t = t.replace(i, j)
    # ====== shortcut ====== #
    shortcut_words = {
        "can't": "cannot",
        "'s": " is",
        "'m": " am",
        "'d": " would",
        "'ve": " have",
        "n't": " not",
        "'re": " are",
        "'ll": " will",
        "nt't": " not"
    }
    for i, j in shortcut_words.iteritems():
        t = t.replace(i, j)
    # ====== other anno ====== #
    t = laugh_regex.sub('', t)
    t = t.strip()
    return t


def collapse_duplicate(conversation, text_func, idx_func,
                       other_func=None):
    it = iter(conversation)
    convs = []
    t = [] # text
    other = [] # other information
    i = -1 # index: who is talking
    while True:
        try:
            _ = it.next()
            # just first start
            if i < 0:
                i = idx_func(_)
            t.append(text_func(_))
            # extract other if necessary
            if other_func is not None:
                other.append(other_func(_))
            # switched turn, different person talking
            if idx_func(_) != i:
                if other_func is None:
                    convs.append((', '.join(t[:-1]), i))
                else:
                    convs.append((', '.join(t[:-1]), other[:-1], i))
                i = idx_func(_)
                t = t[-1:]
                other = other[-1:]
        except StopIteration:
            break
    # add the final sentences
    if len(t) > 0:
        if other_func is None:
            convs.append((','.join(t), i))
        else:
            convs.append((','.join(t), other, i))
    return convs


def is_overlap(s1, e1, s2, e2):
    """True: if (s2,e2) is sufficient overlap on (s1,e1)"""
    if s2 >= s1 and e2 <= e1: # winthin (s1, e1)
        return True
    elif (s2 < s1 and e2 > s1) and (s1 - s2) < (e2 - s1): # s1 within (s2, e2)
        return True
    elif (s2 < e1 and e2 > e1) and (e1 - s2) > (e2 - e1): # e1 within (s2, e2)
        return True
    return False


def get_all_laugh(laugh_anno, start, end):
    return [(i, e - s, idx) for i, s, e, idx in laugh_anno
            if is_overlap(start, end, s, e)]


# ===========================================================================
# processing
# ===========================================================================
# files -> [topic, conversation, laugh_anno, all_laugh, (start_time, end_time)]
#  conversation -> [(text, speaker_id), ...]
#  laugh_anno -> [(laugh_type, duration, speaker_id), ...] # mapping of each sentence to laugh anno
#  all_laugh -> [(laugh_type, duration, speaker_id), ...] # all laugh occured during the topic
# NOTE: len(conversation) = len(laugh_anno)
data = OrderedDict()
for f in files:
    features = []
    for ID, (to, start, end) in enumerate(topics[f]):
        # ====== calibrate some special cases ====== #
        if ID == 0 and to == 'greeting':
            start = 0
        la = laugh[f]
        # ====== extract conversation ====== #
        conversation = [(filter_trans(i), s, e, idx) for i, s, e, idx in trans[f]]
        conversation = [i for i in conversation if len(i[0]) > 0]
        # merge laugh anno
        conversation = [(i, get_all_laugh(la, s, e), idx)
                        for i, s, e, idx in conversation
                        if is_overlap(start, end, s, e)]
        conversation = collapse_duplicate(conversation,
                                          lambda x: x[0], lambda x: x[-1],
                                          lambda x: x[1])
        laugh_anno = [list(chain(*i[1])) for i in conversation]
        conversation = [(i[0], i[-1]) for i in conversation]
        # ====== store ====== #
        features.append((to, conversation, laugh_anno,
                         get_all_laugh(la, start, end),
                         (float(start), float(end))))
    data[f] = features # files -> features
# ====== save ====== #
cPickle.dump(data,
             open(os.path.join(CODE_PATH, 'nlp', '%s_text' % args['ds']), 'w'),
             protocol=cPickle.HIGHEST_PROTOCOL)

# ===========================================================================
# Preprocessing
# ===========================================================================
embedder = F.load_glove(ndim=100)
tokenizer = text.Tokenizer(
    nb_words=1200,
    stopwords=True,
    preprocessors=[text.TransPreprocessor('!\'"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'),
                   text.CasePreprocessor(lower=True)],
    filters=None,
    batch_size=512, nb_threads=4,
    print_progress=False)
all_conversations_and_topics = [i[0] for f, dat in data.iteritems()
                               for d in dat
                               for i in d[1]] + \
    [d[0] for f, dat in data.iteritems() for d in dat]
print('Found:', len(all_conversations_and_topics), 'sentences')
tokenizer.fit(all_conversations_and_topics, vocabulary=embedder)
print('Tokenizer summary:')
for i, j in tokenizer.summary.iteritems():
    print('-', i, ':', j)

# ===========================================================================
# Convert all data to matrix
# ===========================================================================
data_matrix = OrderedDict()
longest_conversation = 0
the_conversation = None
the_topic = None

for f, dat in progbar(data.items()):
    features = []
    for topic, convs, laugh, alllaugh, (start, end) in dat:
        if len(convs) > longest_conversation:
            the_conversation = convs
            the_topic = topic
            longest_conversation = len(convs)
        convs_seq = tokenizer.transform([i[0] for i in convs],
                                        mode='seq', dtype='int32',
                                        padding='pre', truncating='pre', value=0.,
                                        end_document=None, maxlen=None,
                                        token_not_found='ignore')
        convs_tfidf = tokenizer.transform([i[0] for i in convs],
                                        mode='tfidf', dtype='int32',
                                        padding='pre', truncating='pre', value=0.,
                                        end_document=None, maxlen=None,
                                        token_not_found='ignore')
        topic_seq = tokenizer.transform([topic],
                                        mode='seq', dtype='int32',
                                        padding='pre', truncating='pre', value=0.,
                                        end_document=None, maxlen=None,
                                        token_not_found='ignore')
        topic_tfidf = tokenizer.transform([topic],
                                          mode='tfidf', dtype='int32',
                                          padding='pre', truncating='pre', value=0.,
                                          end_document=None, maxlen=None,
                                          token_not_found='ignore')
        # feature contains
        features.append((topic,
                         topic_seq, convs_seq,
                         topic_tfidf, convs_tfidf,
                         laugh, alllaugh,
                         (start, end)))
    data_matrix[f] = features
data_matrix['longest_conversation'] = (longest_conversation, the_conversation, the_topic)
print('Longest conversation:', longest_conversation)
print(the_conversation)
print(the_topic)

# ====== save ====== #
cPickle.dump(tokenizer,
             open(os.path.join(CODE_PATH, 'nlp', '%s_tokenizer' % args['ds']), 'w'),
             protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(data_matrix,
             open(os.path.join(CODE_PATH, 'nlp', '%s_matrix' % args['ds']), 'w'),
             protocol=cPickle.HIGHEST_PROTOCOL)
