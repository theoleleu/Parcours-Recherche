# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Validation and classification
(train)            :  inner-kfold classifier
(train, test)      :  kfold classifier
(train, dev, test) :  split classifier

"""
from __future__ import absolute_import, division, unicode_literals

import logging
import numpy as np
from senteval.tools.classifier import MLP

import sklearn
assert(sklearn.__version__ >= "0.18.0"), \
    "need to update sklearn to version >= 0.18.0"
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from random import *
import torch
from copy import deepcopy
def plot(x):
    i=0
    epochs=len(x[0])
    plt.plot(list(range(i, i+epochs)), x[0], color=(random(),random(),random()))
    plt.show()


def get_classif_name(classifier_config, usepytorch):
    if not usepytorch:
        modelname = 'sklearn-LogReg'
    else:
        nhid = classifier_config['nhid']
        optim = 'adam' if 'optim' not in classifier_config else classifier_config['optim']
        bs = 64 if 'batch_size' not in classifier_config else classifier_config['batch_size']
        modelname = 'pytorch-MLP-nhid%s-%s-bs%s' % (nhid, optim, bs)
    return modelname

# Pytorch version
class InnerKFoldClassifier(object):
    """
    (train) split classifier : InnerKfold.
    """
    def __init__(self, X0, y0, X1, y1, X2, y2, config, clf):
        self.X0,self.y0 = X0,y0
        self.X1,self.y1 = X1,y1
        self.X2,self.y2 = X2,y2
        self.featdim = X0.shape[1]
        self.nclasses = config['nclasses']
        self.seed = config['seed']
        self.devresults = []
        self.testresults = []
        self.usepytorch = config['usepytorch']
        self.classifier_config = config['classifier']
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)
        #self.devothercorp=[]
        self.clf=clf
        if clf!=0:
            self.devothertask=clf.devothertask
            self.precedent_splits=clf.precedent_splits
        else:
            self.devothertask,self.precedent_splits=[[],[],[],[]],[]
        self.k = 5 if 'kfold' not in config else config['kfold']

    def run(self):
        logging.info('Training {0} with (inner) {1}-fold cross-validation'
                     .format(self.modelname, self.k))
        
        regs = [10**t for t in range(-5, -1)] if self.usepytorch else \
               [2**t for t in range(-2, 4, 1)]
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=1111)
        innerskf = StratifiedKFold(n_splits=self.k, shuffle=True,
                                   random_state=1111)
        count = 0
        train_idx0, test_idx0=list(skf.split(self.X0, self.y0))[0]
        train_idx1, test_idx1=list(skf.split(self.X1, self.y1))[0]
        train_idx2, test_idx2=list(skf.split(self.X2, self.y2))[0]
        count += 1
        X_train0, X_test0 = self.X0[train_idx0],self.X0[test_idx0]
        y_train0, y_test0 = self.y0[train_idx0], self.y0[test_idx0]
        X_train1, X_test1 = self.X1[train_idx1],self.X1[test_idx1]
        y_train1, y_test1 = self.y1[train_idx1],self.y1[test_idx1]
        X_train2,X_test2 = self.X2[train_idx2],self.X2[test_idx2]
        y_train2, y_test2 = self.y2[train_idx2],self.y2[test_idx2]
        scores = []
        for reg in regs:
            print(len(regs))
            regscores = []
            isk0=list(innerskf.split(X_train0, y_train0))
            isk1=list(innerskf.split(X_train1, y_train1))
            isk2=list(innerskf.split(X_train2, y_train2))
            n=len(isk0)
                #for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
            inner_train_idx0, inner_test_idx0=isk0[0]
            inner_train_idx1, inner_test_idx1=isk1[0]
            inner_train_idx2, inner_test_idx2=isk2[0]
            X0_in_train,y0_in_train=X_train0[inner_train_idx0],y_train0[inner_train_idx0]
            X1_in_train,y1_in_train=X_train1[inner_train_idx1],y_train1[inner_train_idx1]
            X2_in_train,y2_in_train=X_train2[inner_train_idx2],y_train2[inner_train_idx2]
            X0_in_test=X_train0[inner_test_idx0]
            y0_in_test=y_train0[inner_test_idx0]
            X1_in_test,y1_in_test=X_train1[inner_test_idx1],y_train1[inner_test_idx1]
            X2_in_test,y2_in_test=X_train2[inner_test_idx2],y_train2[inner_test_idx2]
            if self.clf==0:#True:#
                if self.usepytorch:
                    clf = MLP(self.classifier_config, appr=0, inputdim=self.featdim,
                        nclasses=self.nclasses, precedent_splits=self.precedent_splits, 
                        devothertask=self.devothertask, l2reg=reg, seed=self.seed)
                    clf.fit(X0_in_train, y0_in_train,X1_in_train, y1_in_train,X2_in_train, y2_in_train,
                    validation_data=[(X0_in_test, y0_in_test),(X1_in_test, y1_in_test),(X2_in_test, y2_in_test)])
                else:
                    clf = LogisticRegression(C=reg, random_state=self.seed)
                    clf.fit(X0_in_train, y0_in_train)
            else :
                clf = MLP(self.classifier_config, appr=0, inputdim=self.featdim,
                        nclasses=self.nclasses, precedent_splits=self.precedent_splits, 
                        devothertask=self.clf.devothertask, l2reg=reg, seed=self.seed)
                clf.model.load_state_dict(self.clf.model.state_dict())
                clf.fit(X0_in_train, y0_in_train,X1_in_train, y1_in_train,X2_in_train, y2_in_train,
                    validation_data=[(X0_in_test, y0_in_test),(X1_in_test, y1_in_test),(X2_in_test, y2_in_test)])
                    
            regscores.append((clf.score(X0_in_test, y0_in_test)+clf.score(X1_in_test, y1_in_test)+clf.score(X2_in_test, y2_in_test))/3)
            scores.append(round(100*np.mean(regscores), 2))
        scores.append(0.77)
        optreg = 10**(-5)#regs[np.argmax(scores)]
        logging.info('Best param found at split {0}: l2reg = {1} \
            with score {2}'.format(count, optreg, np.max(scores)))
        self.devresults.append(np.max(scores))
        if self.clf==0:#True:#
            if self.usepytorch:
                clf = MLP(self.classifier_config, appr=1, inputdim=self.featdim,
                        nclasses=self.nclasses, precedent_splits=self.precedent_splits,
                        devothertask=self.devothertask, l2reg=optreg, seed=self.seed)

                clf.fit(X_train0, y_train0,X_train1, y_train1,X_train2, y_train2, validation_split=0.05)
            else:
                clf = LogisticRegression(C=optreg, random_state=self.seed)
                clf.fit(X_train0, y_train0)
                print('fail')
        else :
            clf = MLP(self.classifier_config, appr=1, inputdim=self.featdim,
                nclasses=self.nclasses, precedent_splits=self.precedent_splits, 
                devothertask=self.clf.devothertask, l2reg=optreg, seed=self.seed)
            clf.model.load_state_dict(self.clf.model.state_dict())
            clf.fit(X_train0, y_train0,X_train1, y_train1,X_train2, y_train2, validation_split=0.05)
        self.clf=clf
            
        self.testresults.append(round(100*(self.clf.score(X_test0, y_test0)+self.clf.score(X_test1, y_test1)+self.clf.score(X_test2, y_test2))/3, 2))
        print('Test tache 1: ',round(100*(self.clf.score(X_test0, y_test0))))
        print('Test tache 2: ',round(100*(self.clf.score(X_test1, y_test1))))
        print('Test tache 3: ',round(100*(self.clf.score(X_test2, y_test2))))
        devaccuracy = round(np.mean(self.devresults), 2)
        testaccuracy = round(np.mean(self.testresults), 2)
        return devaccuracy, testaccuracy, self.clf

