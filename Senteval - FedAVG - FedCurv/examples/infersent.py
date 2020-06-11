# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
from time import *
import numpy as np
import matplotlib.pyplot as plt
from random import *

# get models.py from InferSent repo
from models import InferSent
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


# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_W2V = 'fasttext/crawl-300d-2M.vec' #'glove/glove.840B.300d.txt'  # or  for V2
MODEL_PATH = 'infersent2.pkl'
V = 2 # version of InferSent

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V)

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)


def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'sgd,lr=0.1', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load InferSent model
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.set_w2v_path(PATH_TO_W2V)
    sleep(10)
    params_senteval['infersent'] = model.cuda()
    sleep(10)
    se = senteval.engine.SE(params_senteval, batcher, prepare)#'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    transfer_tasks = ['PR']
    resultsM, clf = se.eval('PR', 0)
    plot(clf.devothertask)#On retire subj qui ne réalise pas la meme chose
    print(resultsM)
