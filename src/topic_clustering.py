# -*- coding: utf-8 -*-
# PRIMUS - Eta Lite
from __future__ import print_function, division, absolute_import

import re

import numpy as np

from odin.utils import cache_disk, flatten_list, ctext
from odin import fuel as F

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ===========================================================================
# Constants
# ===========================================================================
embedder = F.load_glove(ndim=100)
print("Loaded Embedding:", str(embedder))
_replace_pat = [
    re.compile('\[\.*\]'), # [...]
    re.compile('\(.*\)'), # (laugh)
    re.compile('\*.*\*'), # *laugh*
    re.compile('[a-zA-Z]*-\s?'), # wh-, th-
]


# ===========================================================================
# Helpers
# ===========================================================================
def preprocess_text(text):
    for r in _replace_pat:
        text = r.sub('', text)
    return text


def get_wordnet_pos(treebank_tag):
    # ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


# @cache_disk
def tokenize(docs, keep_stopword=True, keep_oov=True, keep_punct=False,
             lemmatizer=True, stem=True, lower=True):
    # ====== prepare ====== #
    docs = flatten_list(docs)
    # ====== process ====== #
    tk_docs = []
    lemmatiser = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for i, doc in enumerate(docs):
        tokens_pos = nltk.pos_tag(word_tokenize(doc)) # Generate list of tokens
        tokens = []
        for word, pos in tokens_pos:
            if not keep_stopword and word in stopwords.words('english'):
                continue
            if not keep_punct and pos in (',', '.', ":", "(", ")", "[", "]",
                                          '\'', '\"'):
                continue
            if lemmatizer:
                pos = get_wordnet_pos(pos)
                if len(pos) > 0:
                    word = lemmatiser.lemmatize(word, pos)
            if stem:
                word = stemmer.stem(word)
            if lower:
                word = word.lower()
            tokens.append(word)
        tk_docs.append(tokens)
    return tk_docs


# ===========================================================================
# Main
# ===========================================================================
@cache_disk
def _create_model(text, keep_stopword=True, keep_oov=True, keep_punct=False,
                  lemmatizer=True, stem=True, lower=True,
                  nb_topics=6, nb_top_words=12):
    original_text = list(text)
    text = [preprocess_text(t) for t in text]
    text = tokenize(text, keep_stopword=keep_stopword, keep_oov=keep_oov,
                    keep_punct=keep_punct,
                    lemmatizer=lemmatizer, stem=stem, lower=lower)
    vectorizer = TfidfVectorizer(max_df=0.95, max_features=None,
                                 min_df=2, stop_words=None)
    tk_vec = vectorizer.fit_transform([' '.join(i) for i in text])
    terms = vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_components=nb_topics,
                                   max_iter=8,
                                   learning_method='online',
                                   batch_size=len(text),
                                   learning_offset=50.,
                                   random_state=0)
    Y = np.argmax(lda.fit_transform(tk_vec), axis=-1)
    assert len(Y) == len(text) and len(Y) == len(original_text)
    print("\nTopics in LDA model:")
    keywords = {}
    sentences = {}
    for topic_idx, tp in enumerate(lda.components_):
        # store keywords
        topic_terms = [terms[i]
                       for i in tp.argsort()[:-nb_top_words - 1:-1]]
        keywords[topic_idx] = topic_terms
        # store sampled sentences
        samples = [original_text[i]
            for i, y in enumerate(Y) if y == topic_idx]
        samples = np.random.choice(samples, size=min(25, len(samples)),
                                   replace=False)
        sentences[topic_idx] = samples
    # ====== create return results ====== #
    keywords = [val for idx, val in sorted(keywords.items(), key=lambda x: x[0])]
    sentences = [val for idx, val in sorted(sentences.items(), key=lambda x: x[0])]
    return keywords, sentences, \
        {i: j for i, j in zip(original_text, Y)}


def train_topic_clusters(text, keep_stopword=True, keep_oov=True, keep_punct=False,
                         lemmatizer=True, stem=True, lower=True,
                         nb_topics=6, nb_top_words=12,
                         print_log=True):
    keywords, sentences, model = _create_model(text,
            keep_stopword=keep_stopword, keep_oov=keep_oov, keep_punct=keep_punct,
            lemmatizer=lemmatizer, stem=stem, lower=lower,
            nb_topics=nb_topics, nb_top_words=nb_top_words)
    for i in range(nb_topics):
        print(ctext("Topic #%d:", 'yellow') % i,
              " ".join(keywords[i]))
        for j in set(sentences[i]):
            print('\t', j)
    return model
