import random
import torch.contrib
import torch
from torchvision import datasets#gestion des données
from torch.nn import functional as F#fonctions (gradient, cross entropy...)
from torch import nn#neural netwaork
import numpy as np#opérations matricielles
from copy import deepcopy#copie
import matplotlib.pyplot as plt#graphiques
from tqdm import tqdm
#from torch.contrib import SWA
import torch.autograd as autograd
import matplotlib.pyplot as plt#graphiques
import matplotlib.image as mpimg
from IPython import display
import time
import sys
from typing import Dict
from argparse import Namespace
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module

epochs = 50
lr = 1e-3
batch_size = 100
sample_size = 100
hidden_size = 50
num_task = 3


assert int(torch.__version__.split(".")[1]) >= 4, "PyTorch 0.4+ required"

def fim_diag(model: Module, data_loader: list, samples_no: int = None, empirical: bool = False, device: torch.device = None, verbose: bool = False, every_n: int = None) -> Dict[int, Dict[str, Tensor]]:
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    seen_no = 0
    last = 0
    tic = time.time()
    all_fims = dict({})
    i=0
    n=len(data_loader)
    while (samples_no is None or seen_no < samples_no) and i<n:
        data, target = data_loader[i]
        i+=1
        if device is not None:
            data = data.to(device)
            if empirical:
                target = target.to(device)

        logits = model(data)
        if empirical:
            outdx = target.unsqueeze(1)
        else:
            outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
        samples = logits.gather(1, outdx)

        idx, batch_size = 0, data.size(0)
        while idx < batch_size and (samples_no is None or seen_no < samples_no):
            model.zero_grad()
            torch.autograd.backward(samples[idx], retain_graph=True)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    fim[name] += (param.grad * param.grad)
                    fim[name].detach_()
            seen_no += 1
            idx += 1

            if verbose and seen_no % 100 == 0:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
                tic, last = toc, seen_no
                sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.4f} samples/s.")

            if every_n and seen_no % every_n == 0:
                all_fims[seen_no] = {n: f.clone().div_(seen_no).detach_()
                                     for (n, f) in fim.items()}

    if verbose:
        if seen_no > last:
            toc = time.time()
            fps = float(seen_no - last) / (toc - tic)
        sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.5f} samples/s.\n")

    for name, grad2 in fim.items():
        grad2 /= float(seen_no)

    all_fims[seen_no] = fim

    return all_fims
class MLP(nn.Module):
    def __init__(self, hidden_size=5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 4)
    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def penalty(model,data,task):
    nbtasks=len(model)
    pen=0
    means=[]
    for i in range(nbtasks):
      means.append({})
      for n, p in model[i].named_parameters():
        means[i][n] =p.data
    for i in range(nbtasks):
      fim=fim_diag(model[task],data[i])
      fim=fim[400]
      for n, p in model[task].named_parameters():
        if 'weight' in n: 
          dm=means[task][n]-means[i][n]
          if len(dm.size())>1:
            d=dm[:,0]
          fimd=fim[n]
          while len(fimd.size())>1:
            fimd=fimd[0]
          fimd=fimd[0]

        
          pen+= fimd*torch.dot(d,d)
    return pen


def process(epochs,  train_loader : list, dev_loader : list, test_loader : list, importance, use_cuda=True, weight=None):
    model=[]
    loss, dev_loss, acc = [], [], []
    opt = []
    #opt = torch.contrib.optim.SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)  base_
    for task in range(num_task):
        loss.append([])
        dev_loss.append([])
        acc.append([])
        M=MLP(hidden_size)
        model.append(M) 
        opt.append(torch.optim.SGD(M.parameters(), lr=lr))

    for e in tqdm(range(epochs)):
        for task in range(num_task):
            if e==0:
              epoch_loss,epoch_dev_loss=train(model, opt[task], task, train_loader, dev_loader[task], importance,1)
            else:
              epoch_loss,epoch_dev_loss=train(model, opt[task], task, train_loader, dev_loader[task], importance)
            loss[task].append(epoch_loss)
            dev_loss[task].append(epoch_dev_loss)
            for sub_task in range(task + 1):
                task_acc=test(model[task], test_loader[sub_task])
                acc[sub_task].append(task_acc)
        #opt.swap_swa_sgd()
    return loss, dev_loss, acc

def train(model, optimizer, task : int, train_load: list, dev_load: list, importance: float, ini=0):
    model[task].train()
    epoch_loss, dev_epoch_loss = 0, 0
    if ini==0:
        penal=penalty(model,train_load,task)
    else:
        penal=0
    for inp,target in train_load[task]:
        optimizer.zero_grad()
        output = model[task](inp)
        loss = F.cross_entropy(output, target) + importance * penal
        epoch_loss += float(loss.item())

    for inp,target in dev_load:
        optimizer.zero_grad()
        output = model[task](inp)
        dloss = F.cross_entropy(output, target) + importance * penal
        dev_epoch_loss += float(dloss.item())

    for inp,target in train_load[task]:
        optimizer.zero_grad()
        output = model[task](inp)
        loss = F.cross_entropy(output, target) + importance * penal
        loss.backward()
        optimizer.step()
    return epoch_loss / float(len(train_load[task])), dev_epoch_loss / float(len(dev_load))

def test(model: nn.Module, data_loader: list):
    model.eval()
    correct = 0
    for input, target in data_loader:
        output = model(input)
        estimation=output.max(dim=1)[1]
        correct += (estimation == target).data.sum()
    return float(correct.item())/ float(len(data_loader))