# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Pytorch Classifier class in the style of scikit-learn
Classifiers include Logistic Regression and MLP
"""

from __future__ import absolute_import, division, unicode_literals

import numpy as np
from copy import deepcopy
from senteval import utils

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch.autograd as autograd
from torch.distributions import Categorical

def plot(x):
    tot=0
    if len(x)>40:
        for t2, v2 in enumerate(x):
            epochs=len(v2)
            plt.plot(list(range(tot, tot+epochs)), v2)
            tot+=epochs
        plt.show()


def fim_diag(model: nn.Module, data_loader: list):
    model.eval()
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    compt,n = 0,len(data_loader[0])
    #for i in range(n):
    #print(data_loader[0].size())
    data, target = data_loader[0],data_loader[1]
    logits = model(data)
        #print(logits)
    outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
    samples = logits.gather(1, outdx)
    batch_size = data.size(0)
    for j in range(batch_size):
        model.zero_grad()
        torch.autograd.backward(samples[j], retain_graph=True)
        for name, param in model.named_parameters():
            if param.requires_grad:
                fim[name] += (param.grad * param.grad)
                fim[name].detach_()
    compt += batch_size

    for name, grad2 in fim.items():
        grad2 /= float(compt)

    return fim

def penalty(model,data,task):
    nbtasks=len(model)
    u,v={},{}
    means=[]
    for i in range(nbtasks):
        means.append({})
        for n, p in model[i].named_parameters():
            means[i][n] =p.data
    for i in range(nbtasks):
        #print(data[i])
        fim=fim_diag(model[i],data[i])
        d=means[i]
      
        for n, p in fim.items():
            if 'weight' in n:
                if n in u:
                    u[n]+=fim[n]
                    v[n]+=fim[n]*d[n]
                else:
                    u[n]=fim[n]
                    v[n]=fim[n]*d[n]
    return u,v




class PyTorchClassifier(object):
    def __init__(self, inputdim, nclasses, appr, devothertask, l2reg=0., batch_size=64, seed=1111, 
                 cudaEfficient=False):
        # fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.appr=appr
        self.inputdim = inputdim
        self.nclasses = nclasses
        self.l2reg = l2reg
        self.batch_size = batch_size
        self.cudaEfficient = cudaEfficient
        self.devothertask=devothertask
    def prepare_split(self, X0, y0,X1, y1,X2, y2, validation_data=None, validation_split=None):
        # Preparing validation data
        assert validation_split or validation_data
        vec_train=[]
        vec_test=[]

        if validation_data is not None:
            trainX0, trainy0 = X0, y0
            trainX1, trainy1 = X1, y1
            trainX2, trainy2 = X2, y2
            devX0, devy0 = validation_data[0]
            devX1, devy1 = validation_data[1]
            devX2, devy2 = validation_data[2]

        else:
            permutation = np.random.permutation(len(X0))
            trainidx = permutation[int(validation_split * len(X0)):]
            devidx = permutation[0:int(validation_split * len(X0))]
            trainX0, trainy0 = X0[trainidx], y0[trainidx]
            devX0, devy0 = X0[devidx], y0[devidx]

            permutation = np.random.permutation(len(X1))
            trainidx = permutation[int(validation_split * len(X1)):]
            devidx = permutation[0:int(validation_split * len(X1))]
            trainX1, trainy1 = X1[trainidx], y1[trainidx]
            devX1, devy1 = X1[devidx], y1[devidx]

            permutation = np.random.permutation(len(X2))
            trainidx = permutation[int(validation_split * len(X2)):]
            devidx = permutation[0:int(validation_split * len(X2))]
            trainX2, trainy2 = X2[trainidx], y2[trainidx]
            devX2, devy2 = X2[devidx], y2[devidx]

        vec_test.append((devX0, devy0))
        vec_test.append((devX1, devy1))
        vec_test.append((devX2, devy2))
        device = torch.device('cpu') if self.cudaEfficient else torch.device('cuda')
        trainX = torch.from_numpy(trainX0).to(device, dtype=torch.float32)
        trainy = torch.from_numpy(trainy0).to(device, dtype=torch.int64)
        vec_train.append((trainX,trainy))
        trainX = torch.from_numpy(trainX1).to(device, dtype=torch.float32)
        trainy = torch.from_numpy(trainy1).to(device, dtype=torch.int64)
        vec_train.append((trainX,trainy))
        trainX = torch.from_numpy(trainX2).to(device, dtype=torch.float32)
        trainy = torch.from_numpy(trainy2).to(device, dtype=torch.int64)
        vec_train.append((trainX,trainy))
        return vec_train,vec_test

    def fit(self, X0, y0, X1, y1,X2, y2, validation_data=None, validation_split=None,
            early_stop=True):
        self.nepoch = 0
        bestaccuracy = -1
        stop_train = False
        early_stop_count = 0

        # Preparing validation data
        #trainX, trainy, devX, devy = self.prepare_split(X, y, validation_data,
        #                                                validation_split)
        # Training
        vec_train,vec_dev = self.prepare_split(X0,y0,X1,y1,X2,y2, validation_data,
                                                        validation_split)
        n=len(vec_train)
        # Training
        bestaccuracy=0
        beta=1/n
        self.models=[deepcopy(self.model) for i in range(n)]
        self.models0=deepcopy(self.models)
        while not stop_train and self.nepoch <= self.max_epoch:#max epoch a définir
            print(self.nepoch)
            for i in range(n):#Averaging
                self.trainepoch(vec_train, i, epoch_size=self.epoch_size)

            self.models0=deepcopy(self.models)

            if self.appr==1:
                for j,(dX,dY) in enumerate(vec_dev):
                    pacc= self.score(dX, dY)
                    self.devothertask[j].append(pacc)
            self.nepoch+=self.epoch_size
            dic1=deepcopy(self.models[0].state_dict())
            for name1, param1 in self.models[0].named_parameters():
                dic1[name1].data.copy_(dic1[name1].data*beta)
            for i in range(1,n):
                for name, param in self.models[i].named_parameters():
                    if name in dic1:
                        dic1[name].data.copy_(dic1[name].data + beta*param.data)
            self.model.load_state_dict(dic1)
            self.models=[deepcopy(self.model) for i in range(n)]
            accuracy = (self.score(vec_dev[0][0],vec_dev[0][1])+self.score(vec_dev[1][0],vec_dev[1][1]) +self.score(vec_dev[2][0],vec_dev[2][1])) /3
            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                bestmodel = deepcopy(self.model)
            elif early_stop:
                if early_stop_count >= self.tenacity:
                    stop_train = True
                early_stop_count += 1
        return bestaccuracy

    def trainepoch(self, vec_tr, j, epoch_size=1):
        tup=vec_tr[j]
        X,y=tup
        optim_fn, optim_params = utils.get_optimizer(self.optim)
        optimizer= optim_fn(self.models[j].parameters(), **optim_params)
        optimizer.param_groups[0]['weight_decay'] = self.l2reg
        self.models[j].train()
#Partie de pénalisation à faire fonctionner 
#        u,v=penalty(self.models0,vec_tr,j)
        
#        penal=0
#        for n, p in self.models[j].named_parameters():
#          if 'weight' in n:
#            d=p.data
#            un,vn=u[n],v[n]
#            if len(d.size())>1:
#              d=d[:,0]
#              un,vn=un[:,0],vn[:,0]
#        penal=torch.dot(d,un*d)-2*torch.dot(d,vn)

        for _ in range(self.nepoch, self.nepoch + epoch_size):
            permutation = np.random.permutation(len(X))
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.from_numpy(permutation[i:i + self.batch_size]).long().to(X.device)

                Xbatch = X[idx]
                ybatch = y[idx]

                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.models[j](Xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)#+0.01*penal

                all_costs.append(loss.data.item())
                # backward
                optimizer.zero_grad()
                loss.backward()
                # Update parameters
                optimizer.step()
        self.nepoch += epoch_size

    def score(self, devX, devy):
        self.model.eval()
        correct = 0
        if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                ybatch = devy[i:i + self.batch_size]
                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.model(Xbatch)
                pred = output.data.max(1)[1]
                correct += pred.long().eq(ybatch.data.long()).sum().item()
            accuracy = 1.0 * correct / len(devX)
        return accuracy

    def predict(self, devX):
        self.model.eval()
        if not isinstance(devX, torch.cuda.FloatTensor):
            devX = torch.FloatTensor(devX).cuda()
        yhat = np.array([])
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                output = self.model(Xbatch)
                yhat = np.append(yhat,
                                 output.data.max(1)[1].cpu().numpy())
        yhat = np.vstack(yhat)
        return yhat

    def predict_proba(self, devX):
        self.model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                vals = F.softmax(self.model(Xbatch).data.cpu().numpy())
                if not probas:
                    probas = vals
                else:
                    probas = np.concatenate(probas, vals, axis=0)
        return probas


"""
MLP with Pytorch (nhid=0 --> Logistic Regression)
"""

class MLP(PyTorchClassifier):
    def __init__(self, params, inputdim, nclasses, appr, precedent_splits, devothertask,l2reg=0., batch_size=64,
                 seed=1111, cudaEfficient=False):
        super(self.__class__, self).__init__(inputdim, nclasses, appr, devothertask, l2reg,
                                             batch_size, seed, cudaEfficient)
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """

        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.optim = "adam" if "optim" not in params else params["optim"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 4 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
        self.dropout = 0. if "dropout" not in params else params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else params["batch_size"]
        if params["nhid"] == 0:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),
            ).cuda()
        else:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, params["nhid"]),
                nn.Dropout(p=self.dropout),
                nn.Sigmoid(),
                nn.Linear(params["nhid"], self.nclasses),
            ).cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False

        optim_fn, optim_params = utils.get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg
