# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Binary classifier and corresponding datasets : MR, CR, SUBJ, MPQA
'''
from __future__ import absolute_import, division, unicode_literals

import io
import os
import numpy as np
import logging

from senteval.tools.validation import InnerKFoldClassifier


class BinaryClassifierEval(object):
    def __init__(self, pos, neg, clf,  seed=1111):
        self.seed = seed
        #pos2 = [list(ss_elt) for elt in pos for ss_elt in zip(*elt)]
        #neg2 = [list(ss_elt) for elt in neg for ss_elt in zip(*elt)]
        #self.samples, self.labels = pos2+neg2, [1] * len(pos2) + [0] * len(neg2)
        #self.n_samples = len(self.samples)

        self.samples0, self.labels0 = pos[0]+neg[0], [1] * len(pos[0]) + [0] * len(neg[0])
        self.samples1, self.labels1 = pos[1]+neg[1], [1] * len(pos[1]) + [0] * len(neg[1])
        self.samples2, self.labels2 = pos[2]+neg[2], [1] * len(pos[2]) + [0] * len(neg[2])
        self.n_s1=len(pos[0]+neg[0])
        self.n_s2= len(pos[1]+neg[1])
        self.n_s3=len(pos[2]+neg[2])
        self.clf=clf
    def do_prepare(self, params, prepare):
        # prepare is given the whole text
        return prepare(params, self.samples0+self.samples1+self.samples2)
        # prepare puts everything it outputs in "params" : params.word2id etc
        # Those output will be further used by "batcher".

    def loadFile(self, fpath):
        with io.open(fpath, 'r', encoding='latin-1') as f:
            return [line.split() for line in f.read().splitlines()]

    def run(self, params, batcher):
        
        logging.info('Generating sentence embeddings')
        enc_input0 = []
        sorted_corpus0 = sorted(zip(self.samples0, self.labels0),
                               key=lambda z: (len(z[0]), z[1]))
        sorted_samples0 = [x for (x, y) in sorted_corpus0]
        sorted_labels0 = [y for (x, y) in sorted_corpus0]
        for ii in range(0, self.n_s1, params.batch_size):
            batch0 = sorted_samples0[ii:ii + params.batch_size]
            embeddings0 = batcher(params, batch0)
            enc_input0.append(embeddings0)
        enc_input0 = np.vstack(enc_input0)

        enc_input1 = []
        sorted_corpus1 = sorted(zip(self.samples1, self.labels1),
                               key=lambda z: (len(z[0]), z[1]))
        sorted_samples1 = [x for (x, y) in sorted_corpus1]
        sorted_labels1 = [y for (x, y) in sorted_corpus1]
        for ii in range(0, self.n_s2, params.batch_size):
            batch1 = sorted_samples1[ii:ii + params.batch_size]
            embeddings1 = batcher(params, batch1)
            enc_input1.append(embeddings1)
        enc_input1 = np.vstack(enc_input1)

        enc_input2 = []
        sorted_corpus2 = sorted(zip(self.samples2, self.labels2),
                               key=lambda z: (len(z[0]), z[1]))
        sorted_samples2 = [x for (x, y) in sorted_corpus2]
        sorted_labels2 = [y for (x, y) in sorted_corpus2]
        for ii in range(0, self.n_s3, params.batch_size):
            batch2 = sorted_samples2[ii:ii + params.batch_size]
            embeddings2 = batcher(params, batch2)
            enc_input2.append(embeddings2)
        enc_input2 = np.vstack(enc_input2)

        
        logging.info('Generated sentence embeddings')

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        clef = InnerKFoldClassifier( 
        enc_input0, np.array(sorted_labels0),
        enc_input1, np.array(sorted_labels1),
        enc_input2, np.array(sorted_labels2),config,self.clf)
        devacc, testacc, c = clef.run()
        logging.debug('Dev acc : {0} Test acc : {1}\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc}, c

class CREval(BinaryClassifierEval):
    def __init__(self, task_path, clf, seed=1111):
        logging.debug('***** Transfer task : CR *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'custrev.pos'))
        neg = self.loadFile(os.path.join(task_path, 'custrev.neg'))
        super(self.__class__, self).__init__(pos, neg, clf, seed)

class PREval(BinaryClassifierEval):
    def __init__(self, task_path1, task_path2, task_path3, task_path4, clf, seed=1111):
        logging.debug('***** Transfer task : PR *****\n\n')
        pos1 = self.loadFile(os.path.join(task_path1, 'custrev.pos'))
        neg1 = self.loadFile(os.path.join(task_path1, 'custrev.neg'))
        pos2 = self.loadFile(os.path.join(task_path2, 'rt-polarity.pos'))
        neg2 = self.loadFile(os.path.join(task_path2, 'rt-polarity.neg'))
        pos3 = self.loadFile(os.path.join(task_path3, 'mpqa.pos'))
        neg3 = self.loadFile(os.path.join(task_path3, 'mpqa.neg'))
        #pos4 = self.loadFile(os.path.join(task_path4, 'subj.objective'))
        #neg4 = self.loadFile(os.path.join(task_path4, 'subj.subjective'))
        neg=[neg1,neg2,neg3]
        pos=[pos1,pos2,pos3]
        super(self.__class__, self).__init__(pos, neg, clf, seed)

class MREval(BinaryClassifierEval):
    def __init__(self, task_path, clf, seed=1111):
        logging.debug('***** Transfer task : MR *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'rt-polarity.pos'))
        neg = self.loadFile(os.path.join(task_path, 'rt-polarity.neg'))
        super(self.__class__, self).__init__(pos, neg, clf, seed)


class SUBJEval(BinaryClassifierEval):
    def __init__(self, task_path, clf, seed=1111):
        logging.debug('***** Transfer task : SUBJ *****\n\n')
        obj = self.loadFile(os.path.join(task_path, 'subj.objective'))
        subj = self.loadFile(os.path.join(task_path, 'subj.subjective'))
        super(self.__class__, self).__init__(obj, subj, clf, seed)


class MPQAEval(BinaryClassifierEval):
    def __init__(self, task_path, clf, seed=1111):
        logging.debug('***** Transfer task : MPQA *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'mpqa.pos'))
        neg = self.loadFile(os.path.join(task_path, 'mpqa.neg'))
        super(self.__class__, self).__init__(pos, neg, clf, seed)
