import random
import torch.contrib
import torch
from torch.nn import functional as F#fonctions (gradient, cross entropy...)
from torch import nn#neural netwaork
import numpy as np#opérations matricielles
import matplotlib.pyplot as plt#graphiques
from tqdm import tqdm
from copy import deepcopy
import torch.autograd as autograd
from torch.distributions import Categorical

epochs = 50
lr = 1e-3
batch_size = 100
sample_size = 100
hidden_size = 50
num_task = 3

assert int(torch.__version__.split(".")[1]) >= 4, "PyTorch 0.4+ required"

def fim_diag(model: nn.Module, data_loader: list):
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    compt,n = 0,len(data_loader)
    for i in range(n):
        data, target = data_loader[i]
        logits = model(data)
        outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
        samples = logits.gather(1, outdx)
        j, batch_size = 0, data.size(0)
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
    u,v={},{}
    means=[]
    for i in range(nbtasks):
        means.append({})
        for n, p in model[i].named_parameters():
            means[i][n] =p.data
    for i in range(nbtasks):
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


def process(epochs,  train_loader : list, dev_loader : list, test_loader : list, importance, use_cuda=True, weight=None):
    model, loss, dev_loss, acc, opt = [], [], [], [], []
    for task in range(num_task):
        loss.append([])
        dev_loss.append([])
        acc.append([])
        M=MLP(hidden_size)
        model.append(M) 
        opt.append(torch.optim.SGD(M.parameters(), lr=lr))
    dic1=deepcopy(model[0].state_dict())
    beta = 1/num_task
    for e in tqdm(range(epochs)):
        for task in range(num_task):
            if e==0:
                epoch_loss,epoch_dev_loss=train(model, dic1, opt[task], task, train_loader, dev_loader[task], importance,1)
            else:
                epoch_loss,epoch_dev_loss=train(model, dic1, opt[task], task, train_loader, dev_loader[task], importance)
            loss[task].append(epoch_loss)
            dev_loss[task].append(epoch_dev_loss)
            for sub_task in range(task + 1):
                task_acc=test(model[task], test_loader[sub_task])
                acc[sub_task].append(task_acc)
        
        #averaging
        avg=MLP(hidden_size)
        dic1=deepcopy(model[0].state_dict())
        for name1, param1 in model[0].named_parameters():
            dic1[name1].data.copy_(dic1[name1].data*beta)
        for i in range(1,num_task):
            for name, param in model[i].named_parameters():
                if name in dic1:
                    dic1[name].data.copy_(dic1[name].data + beta*param.data)

    return loss, dev_loss, acc

def train(model, dic1, optimizer, task : int, train_load: list, dev_load: list, importance: float, ini=0):
    model[task].train()
    epoch_loss, dev_epoch_loss = 0, 0
    if ini==0:
        u,v=penalty(model,train_load,task)
    else:
        u,v=0,0
        
    penal=0
    for n, p in model[task].named_parameters():
          if ini==0 and 'weight' in n:
              d=p.data
              un,vn=u[n],v[n]
              if len(d.size())>1:
                  d=d[:,0]
                  un,vn=un[:,0],vn[:,0]
              penal+=torch.dot(d,un*d)-2*torch.dot(d,vn)

    if ini==0:
        model[task].load_state_dict(dic1)#average taken

    #for inp,target in train_load[task]:
    #    output = model[task](inp)
    #    loss = F.cross_entropy(output, target) + importance * penal

    for inp,target in dev_load:
        output = model[task](inp)
        dloss = F.cross_entropy(output, target) + importance * penal
        dev_epoch_loss += float(dloss.item())

    for inp,target in train_load[task]:

        penal=0
        for n, p in model[task].named_parameters():
            if ini==0 and 'weight' in n:
                d=p.data
                un,vn=u[n],v[n]
                if len(d.size())>1:
                    d=d[:,0]
                    un,vn=un[:,0],vn[:,0]
                penal+=torch.dot(d,un*d)-2*torch.dot(d,vn)

        optimizer.zero_grad()
        output = model[task](inp)
        loss = F.cross_entropy(output, target) + importance * penal
        epoch_loss += float(loss.item())
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