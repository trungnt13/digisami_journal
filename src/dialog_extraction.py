# ===========================================================================
# investigate the possibility of modeling conversational topics with low amount of data
# correlation of multi-modal of data with the topic
# automatic topic detection and switch
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
import time
from itertools import chain
from six.moves import cPickle
from collections import OrderedDict

from extract_annos import get_annotation
from odin.preprocessing import text
from odin.utils import UnitTimer
from odin import fuel as F

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# ===========================================================================
# Load data
# ===========================================================================
path = '/mnt/sdb1/digisami_data'
ds = F.Dataset('/home/trung/data/estonian_audio', read_only=True)

trans = get_annotation(path, 'estonian', 'trans', 'eng',
    conversation=True, discrete_time=0.01)
topics = get_annotation(path, 'estonian', 'topic',
    discrete_time=0.01)

# ===========================================================================
# Preprocessing
# ===========================================================================
embedder_wiki = F.load_glove(ndim=50)
embedder_twitter = F.MmapDict('/home/trung/data/glove.twitter.50d',
                              read_only=True)
tokenizer = text.Tokenizer(
    char_level=False,
    nb_words=None,
    stopwords=True,
    preprocessors=[
        text.CasePreprocessor(lower=False)
    ],
    filters=None,
    batch_size=512, nb_processors=4,
    print_progress=True)

for name in trans.keys():
    if '*' in name or name == 'header': continue

convs = trans['C_21_MM_21_22']
topic = topics['C_21_MM_21_22']
train = [i[-1] for i in convs]
# ====== LDA ====== #
tfidf = TfidfVectorizer(max_df=0.95, min_df=2,
                        max_features=None,
                        stop_words=None,
                        lowercase=False)
tfidf_train = tfidf.fit_transform(train)

# Use tf (raw term count) features for LDA.
tf = CountVectorizer(max_df=0.95, min_df=2,
                     max_features=None,
                     stop_words=None,
                     lowercase=False)
tf_train = tf.fit_transform(train)

lda = LatentDirichletAllocation(n_topics=len(topic),
    max_iter=5,
    learning_method='online',
    learning_offset=50.,
    random_state=0)
lda.fit(tf_train)

print("\nTopics in LDA model:")
feature_names = tf.get_feature_names()
nb_top_words = 10
for topic_idx, tp in enumerate(lda.components_):
    print("Topic #%d:" % topic_idx, " ".join([feature_names[i]
                    for i in tp.argsort()[:-nb_top_words - 1:-1]]))

print("Actual topic:")
for t in topic:
    print('*', t[-1])
