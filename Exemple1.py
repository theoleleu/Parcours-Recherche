import random
import torch
from torchvision import datasets#gestion des données
from torch.nn import functional as F#fonctions (gradient, cross entropy...)
from tqdm import tqdm
from torch import nn#neural netwaork
import numpy as np#opérations matricirlles
from copy import deepcopy#copie
from torch.autograd import Variable
import matplotlib.pyplot as plt#graphiques
import matplotlib.image as mpimg
from IPython import display

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



def standard_process(epochs, use_cuda=True, weight=True):
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
        if task == 0 and weight:
            weight = model.state_dict()
    return loss, dev_loss, acc, weight
    
    
def normal_train(model, optimizer, data_load: list,dev_load : list):
    model.train()
    epoch_loss,epoch_dev_loss = 0, 0
    for inp,target in data_load:
        optimizer.zero_grad()
        output = model(inp)
        loss = F.cross_entropy(output, target)
        epoch_loss += float(loss.item())
        loss.backward()
        optimizer.step()
    for inp,target in dev_load:
      optimizer.zero_grad()
      output = model(inp)
      devloss = F.cross_entropy(output, target)
      epoch_dev_loss += float(devloss.item())
    return epoch_loss / len(data_load), epoch_dev_loss / len(dev_load)

def test(model: nn.Module, data_loader: list):
    model.eval()
    correct = 0
    for input, target in data_loader:
        output = model(input)
        estimation=F.softmax(output, dim=1).max(dim=1)[1]
        correct += (estimation == target).data.sum()
    return correct/ len(data_loader)

#Test

















#EWC
def ewc_process(epochs, importance, use_cuda=True, weight=None):
    model = MLP(hidden_size)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    loss, acc, ewc = {}, {}, {}
    for task in range(num_task):
        loss[task], dev_loss[task], acc[task] = [], [], []
        if task == 0:
                for _ in tqdm(range(epochs)):
                    epoch_loss,epoch_dev_loss=normal_train(model, optimizer, train_loader[task],dev_loader[task])
                    loss[task].append(epoch_loss)
                    dev_loss[task].append(epoch_dev_loss)
                    acc[task].append(test(model, test_loader[task]))#calcul de l'accuracy
        else:
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
    return epoch_loss / len(data_load), dev_epoch_loss / len(dev_load)

class Model(object):
    def __init__(self, model: nn.Module, dataset: list):#le model d'hyperparamètre et les données
        self.model = model
        self.dataset = dataset
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._fisher = self._diag_fisher()
        for n, p in deepcopy(self.params).items():
            self._means[n] =p.data
    def _diag_fisher(self):
        fisher = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            fisher[n] = p.data
        self.model.eval()
        for input,target in self.dataset:
            self.model.zero_grad()
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label) 
            loss.backward()
            for n, p in self.model.named_parameters():
                fisher[n].data += p.grad.data ** 2 / len(self.dataset)
        fisher = {n: p for n, p in fisher.items()}#Copie du dictionnaire Utilité ?
        return fisher
        
        #Carré du gradient de la log vraisemblance / nbdonnées p.grad.data dérivée de la log vraisemblance car p.data est le delta de la negative log vraisemblance  d'où le carrée de la norme 2 du gradient de la negative log vraisemblance
          
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


#Exemple :
from torch.utils.data import Dataset
size = 1000
size2= 500
sigma = 0.6
a1=np.concatenate((np.random.normal(3, sigma, size), np.random.normal(4, sigma, size), np.random.normal(2, sigma, size),np.random.normal(5, sigma, size)))
a2=np.concatenate((np.random.normal(4, sigma, size), np.random.normal(1.6, sigma, size), np.random.normal(5, sigma, size),np.random.normal(5, sigma, size)))
b1=np.concatenate((np.random.normal(2.8, sigma, size), np.random.normal(3.7, sigma, size), np.random.normal(1.7, sigma, size),np.random.normal(5.1, sigma, size)))
b2=np.concatenate((np.random.normal(3.9, sigma, size), np.random.normal(2.1, sigma, size), np.random.normal(4.9, sigma, size),np.random.normal(5.2, sigma, size)))
c1=np.concatenate((np.random.normal(3.1, sigma, size), np.random.normal(3.7, sigma, size), np.random.normal(1.4, sigma, size),np.random.normal(4.9, sigma, size)))
c2=np.concatenate((np.random.normal(3.9, sigma, size), np.random.normal(1.5, sigma, size), np.random.normal(5.3, sigma, size),np.random.normal(4.7, sigma, size)))
t1=np.concatenate((np.random.normal(3.1, sigma, size2), np.random.normal(3.8, sigma, size2), np.random.normal(1.4, sigma, size2),np.random.normal(5, sigma, size2)))
t2=np.concatenate((np.random.normal(3.9, sigma, size2), np.random.normal(1.7, sigma, size2), np.random.normal(5.7, sigma, size2),np.random.normal(4.8, sigma, size2)))

TR1=[[[i[0],i[1]]] for i in zip(a1,a2)]
TR2=[[[i[0],i[1]]] for i in zip(b1,b2)]
TR3=[[[i[0],i[1]]] for i in zip(c1,c2)]
TE1=[[[i[0],i[1]]] for i in zip(t1,t2)]
TE2=[[[i[0],i[1]]] for i in zip(t1,t2)]
TE3=[[[i[0],i[1]]] for i in zip(t1,t2)]

train_loader,test_loader,dev_loader=[0,0,0],[0,0,0],[0,0,0]
color2=[0]*500+[1]*500+[2]*500+[3]*500
color=[0]*1000+[1]*1000+[2]*1000+[3]*1000
train_loader[0] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TR1,color)]
train_loader[1] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TR2,color)]
train_loader[2] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TR3,color)]
test_loader[0] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TE1,color2)]
test_loader[1] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TE2,color2)]
test_loader[2] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TE3,color2)]
dev_loader[0] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TE1,color2)]
dev_loader[1] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TE2,color2)]
dev_loader[2] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TE3,color2)]

##TESTS
epochs = 50
lr = 1e-3
batch_size = 100
sample_size = 100
hidden_size = 80
num_task = 3
loss, dev_loss, acc, weight = standard_process(epochs)
loss_plot2(loss,dev_loss)
#accuracy_plot(acc)
#loss_ewc, dev_loss_ewc, acc_ewc = ewc_process(epochs, importance=1000)
#loss_plot(loss_ewc)
#accuracy_plot(acc_ewc)
#plt.plot(acc[0], label="sgd")
#plt.plot(acc_ewc[0], label="ewc")
#plt.legend()







