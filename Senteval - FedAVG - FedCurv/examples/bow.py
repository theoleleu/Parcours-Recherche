# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import matplotlib.pyplot as plt
from random import *

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'
# PATH_TO_VEC = 'glove/glove.840B.300d.txt'
PATH_TO_VEC = 'fasttext/crawl-300d-2M.vec'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
def plot(x):
    i=0
    n=len(x)
    for j in range(0,n):
        a=len(x[0])-len(x[j])
        epochs=len(x[j])
        plt.plot(list(range(a+i, a+i+epochs)), x[j], color=(random(),random(),random()))
    plt.show()

def plotj(x,j):
    i=0
    epochs=len(x[j])
    plt.plot(list(range(i, i+epochs)), x[j], color=(random(),random(),random()))
    plt.show()
                  
# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings



# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'sgd,lr=0.1', 'batch_size': 128,
                                 'tenacity': 4, 'epoch_size': 2}


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

def coucou():
    se = senteval.engine.SE(params_senteval, batcher, prepare)#'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ']
    resultsM, clf = se.eval('PR', 0)
    plot(clf.devothertask)
    return clf

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)#'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    transfer_tasks = ['PR']
    resultsM, clf = se.eval('PR', 0)
    plot(clf.devothertask)#On retire subj qui ne réalise pas la meme chose
    print(resultsM)


