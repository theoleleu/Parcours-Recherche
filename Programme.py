import random
import torch
from torchvision import datasets#gestion des données
from torch.nn import functional as F#fonctions (gradient, cross entropy...)
from torch import nn#neural netwaork
import numpy as np#opérations matricielles
from copy import deepcopy#copie
import matplotlib.pyplot as plt#graphiques
from tqdm import tqdm

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



def standard_process(epochs, train_loader : list, dev_loader : list, test_loader : list, use_cuda=True, weight=True):
    model = MLP(hidden_size)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    loss, dev_loss, acc = {}, {}, {}
    for task in range(num_task):
        loss[task] = []
        dev_loss[task] = []
        acc[task] = []
        for _ in tqdm(range(epochs)):
            epoch_loss,epoch_dev_loss=normal_train(model, optimizer, train_loader[task],dev_loader[task])
            loss[task].append(epoch_loss)
            dev_loss[task].append(epoch_dev_loss)
            for sub_task in range(task + 1):
                acc[sub_task].append(test(model, test_loader[sub_task]))
    return loss, dev_loss, acc, weight
    
    
def normal_train(model, optimizer, data_load: list,dev_load : list):
    model.train()
    epoch_loss,epoch_dev_loss = 0, 0
    for inp,target in dev_load:
        optimizer.zero_grad()
        output = model(inp)
        devloss = F.cross_entropy(output, target)
        epoch_dev_loss += float(devloss.item())
    for inp,target in data_load:
        optimizer.zero_grad()
        output = model(inp)
        loss = F.cross_entropy(output, target)
        epoch_loss += float(loss.item())
        loss.backward()
        optimizer.step()
    return epoch_loss / float(len(data_load)), epoch_dev_loss / float(len(dev_load))

def test(model: nn.Module, data_loader: list):
    model.eval()
    correct = 0
    for input, target in data_loader:
        output = model(input)
        estimation=output.max(dim=1)[1]
        correct += (estimation == target).data.sum()
    return float(correct.item())/ float(len(data_loader))









#EWC
class Model(object):
    def __init__(self, model: nn.Module, dataset: list):#le model d'hyperparamètre et les données
        self.model = model
        self.dataset = dataset
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._fisher = self._diag_fisher()
        for n, p in self.params.items():
            self._means[n] =p.data
    def _diag_fisher(self):
        fisher = {}
        for n, p in self.params.items():
            p.data.zero_()
            fisher[n] = p.data
        self.model.eval()
        for input,target in self.dataset:
            self.model.zero_grad()
            output = self.model(input)
            #loss ?????
            loss.backward()
            for n, p in self.model.named_parameters():
                fisher[n].data += p.grad.data ** 2 / len(self.dataset)#Carré du gradient de la log vraisemblance / nbdonnées
        fisher = {n: p for n, p in fisher.items()}#Copie du dictionnaire Utilité ?
        return fisher
        
        #Carré du gradient de la log vraisemblance / nbdonnées p.grad.data dérivée de la log vraisemblance car p.data est le delta de la negative log vraisemblance  d'où le carrée de la norme 2 du gradient de la negative log vraisemblance
    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():#tache n et poids p
            _loss = self._fisher[n] * (p - self._means[n]) ** 2#Pénalisation par information de Fisher
            loss += _loss.sum()#Somme des pénalisations par information de Fisher
        return loss     
         
def ewc_process(epochs, importance, train_loader : list, dev_loader : list, test_loader : list, use_cuda=True, weight=None):
    model = MLP(hidden_size)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    loss, dev_loss, acc, ewc = {}, {}, {}, {}
    for task in range(num_task):
        loss[task], dev_loss[task], acc[task] = [], [], []
        old_tasks = []
        for sub_task in range(task):
            old_tasks = old_tasks + train_loader[sub_task]   
            #old_tasks = random.sample(old_tasks, k=sample_size)# Ou on choisit une fraction des tâches
        for _ in tqdm(range(epochs)):
            epoch_loss,epoch_dev_loss=ewc_train(model, optimizer, train_loader[task], dev_loader[task], Model(model, old_tasks), importance)
            loss[task].append(epoch_loss)
            dev_loss[task].append(epoch_dev_loss)
            for sub_task in range(task + 1):
                task_acc=test(model, test_loader[sub_task])
                acc[sub_task].append(task_acc)
    return loss, dev_loss, acc


def ewc_train(model, optimizer, data_load: list, dev_load: list,ewc: Model, importance: float):
    model.train()
    epoch_loss, dev_epoch_loss = 0, 0
    for input, target in data_load:
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
        epoch_loss += float(loss.item())
        loss.backward()
        optimizer.step()
    for input, target in dev_load:
        optimizer.zero_grad()
        output = model(input)
        dloss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
        dev_epoch_loss += float(dloss.item())#Perte cumulée
        dloss.backward()
        optimizer.step()
    return epoch_loss / float(len(data_load)), dev_epoch_loss / float(len(dev_load))

 
def loss_plot(x):
    for t, v in x.items():
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)
        
def loss_plot2(x,y):
    for t, v in x.items():
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)
    for t, v in y.items():
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)

def accuracy_plot(x):
    for t, v in x.items():
        plt.plot(list(range(t * epochs, num_task * epochs)), v)
    plt.ylim(0, 1)

