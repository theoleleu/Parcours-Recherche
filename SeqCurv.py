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
from torch.contrib import SWA
import torch.autograd as autograd

epochs = 50
lr = 1e-3
batch_size = 100
sample_size = 100
hidden_size = 50
num_task = 3


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


def eval_hessian(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian.cpu().data.numpy()

def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)


def penalty(model,data,task,optimizer):
    model.eval()
    nbtasks=len(model)
    pen=0
    means={}
    for i in range(nbtasks):
      for n, p in model.params.items():
        means[n] =p.data
    for i in range(nbtasks):
      loss=0
      sample=random.sample(data[i], k=sample_size)
      for inp,target in sample:
        optimizer.zero_grad()
        output = model(inp)
        loss += F.cross_entropy(output, target)
      pen+= torch.transpose(means[task]-means[i])*hessian(loss, model[i])*(means[task]-means[i])
    return pen


def process(epochs,  train_loader : list, dev_loader : list, test_loader : list, importance, use_cuda=True, weight=None):
    model=[]
    loss, dev_loss, acc, ewc = {}, {}, {}, {}
    loss[task], dev_loss[task], acc[task] = [], [], []
    base_opt = torch.optim.SGD(model.parameters(), lr=lr)
    opt = torch.contrib.optim.SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
    for task in range(num_task):
        model.append(MLP(hidden_size)) 
    for e in tqdm(range(epochs)):
        for task in range(num_task):
            if e==0:
              epoch_loss,epoch_dev_loss=train(model, opt, task, train_loader, dev_loader[task], importance,1)
            else:
              epoch_loss,epoch_dev_loss=train(model, opt, task, train_loader, dev_loader[task], importance)
            loss[task].append(epoch_loss)
            dev_loss[task].append(epoch_dev_loss)
            for sub_task in range(task + 1):
                task_acc=test(model[task], test_loader[sub_task])
                acc[sub_task].append(task_acc)
        opt.swap_swa_sgd()
    return loss, dev_loss, acc

def train(model, optimizer, task : int, train_load: list, dev_load: list, importance: float, ini=0):
    model[task].train()
    epoch_loss, dev_epoch_loss = 0, 0
    if ini==0:
        penal=penalty(model,train_load,task,optimizer)
    else:
        penal=0
    for input, target in train_load[task]:
        optimizer.zero_grad()
        output = model[task](input)
        loss = F.cross_entropy(output, target) + importance * penal
        epoch_loss += float(loss.item())
        loss.backward()
        optimizer.step()
    for input, target in dev_load:
        optimizer.zero_grad()
        output = model[task](input)
        dloss = F.cross_entropy(output, target) + importance * penal
        dev_epoch_loss += float(dloss.item())#Perte cumulée
        dloss.backward()
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