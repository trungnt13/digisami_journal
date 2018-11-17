"""
Stemming:
    am, are, is => be
    car, cars, car's, cars' => car

"""
from __future__ import print_function, division, absolute_import
from time import time
import cPickle

import numpy as np

from itertools import chain
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import spacy
nlp = spacy.load('en')

estonia_data = cPickle.load(open('all_data', 'r'))
n_features = 256
n_topics = 5
n_top_words = 12


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def text_normalization(text):
    tokens = []
    for t in nlp(text):
        t = t.lemma_.lower().strip() if t.lemma_ != "-PRON-" else t.lower_
        tokens.append(t)
    return ' '.join(tokens)

# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
data_samples = [text_normalization(i[0])
                for i in list(chain(*estonia_data['topics'].values()))]
print(len(data_samples))

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data_samples)

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(data_samples)

# Fit the NMF model
print("Fitting the NMF model with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (len(data_samples), n_features))
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)

print("\n==> Topics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (len(data_samples), n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=10,
                                batch_size=128,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)

print("\n==> Topics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

# ===========================================================================
# Make prediction
# ===========================================================================
topics_labels = {}
for fname, topics in estonia_data['topics'].iteritems():
    topics_labels[fname] = [
        (np.argmax(lda.transform(
            tf_vectorizer.transform([text_normalization(text)]))),
            start, end)
        for text, start, end in topics
    ]
cPickle.dump(topics_labels, open('topic_labels', 'w'),
             protocol=cPickle.HIGHEST_PROTOCOL)

# ===========================================================================
# Visualization
# ===========================================================================
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import colors
from odin import visual
import seaborn

topic_colors = {
    0: 'blue',
    1: 'green',
    2: 'red',
    3: 'cyan',
    4: 'yellow',
    -1: 'white',
}
topic_colors = {i: colors.hex2color(colors.cnames[j])
                for i, j in topic_colors.iteritems()}
laugh_colors = defaultdict(lambda *arg, **kwargs: colors.hex2color(colors.cnames['blue']))


def plot_multiple_images(name, x1, x2, x3):
    # fig.subplots_adjust(hspace=.1, wspace=.1)
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    ax1.set_title(name)

    ax1.imshow(x1)
    ax1.grid(False)
    # ax1.set_frame_on(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylabel('Topic', fontsize=16)

    ax2.imshow(x2)
    ax2.grid(False)
    # ax2.set_frame_on(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylabel('Speaker1', fontsize=16)

    ax3.imshow(x3)
    ax3.grid(False)
    # ax3.set_frame_on(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_ylabel('Speaker2', fontsize=16)

    visual.plot_save('/Users/trungnt13/tmp/%s.pdf' % name, dpi=400, tight_plot=True)


def plot_multiple_images1(name, x1, x2, x3):
    plt.title(name)
    X = np.concatenate((x1, x2, x3), axis=0)
    ax = plt.gca()
    ax.imshow(X)
    ax.grid(False)
    ax.set_frame_on(True)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_ylabel('Speaker2, Speaker1, Topic', fontsize=12)
    visual.plot_save('/Users/trungnt13/tmp/%s.pdf' % name, dpi=400, tight_plot=True)


def draw_rgb(X, colors, length):
    rgb = np.full(shape=(128, 1000, 3), fill_value=0.98)
    for x, start, end in X:
        start = np.floor(start / length * 1000)
        end = np.floor(end / length * 1000)
        x = colors[x]
        rgb[:, start:end, :] = x
    return rgb

for fname, topics in topics_labels.iteritems():
    print(fname)
    laughter = estonia_data['laughter'][fname]
    length = np.ceil(max(topics[-1][2], laughter[-1][2]))
    laughter0 = [i[:-1] for i in laughter if i[-1] == 0]
    laughter1 = [i[:-1] for i in laughter if i[-1] == 1]
    topic_rgb = draw_rgb(topics, topic_colors, length)
    laughter0_rgb = draw_rgb(laughter0, laugh_colors, length)
    laughter1_rgb = draw_rgb(laughter1, laugh_colors, length)
    plot_multiple_images1(fname, topic_rgb, laughter0_rgb, laughter1_rgb)
