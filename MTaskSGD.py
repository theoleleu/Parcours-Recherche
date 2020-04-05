import random
import torch
from torchvision import datasets#gestion des données
from torch.nn import functional as F#fonctions (gradient, cross entropy...)
from torch import nn#neural netwaork
import numpy as np#opérations matricielles
from copy import deepcopy#copie
import matplotlib.pyplot as plt#graphiques
from tqdm import tqdm

epochs = 30
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



def process(epochs, train_loader : list, dev_loader : list, test_loader : list, use_cuda=True, weight=True):
    model = MLP(hidden_size)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    loss, dev_loss, acc = [], [], []
    for task in range(num_task):
        
        loss.append([])
        dev_loss.append([])
        acc.append([])

        for _ in tqdm(range(epochs)):

            epoch_loss,epoch_dev_loss, model=train(model, optimizer, train_loader[task],dev_loader[task])
            loss[task].append(epoch_loss)
            dev_loss[task].append(epoch_dev_loss)

            for sub_task in range(task + 1):

                acc[sub_task].append(test(model, test_loader[sub_task]))
    
    return loss, dev_loss, acc
    
    
def train(model, optimizer, data_load: list,dev_load : list):
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
    for inp,target in data_load:
        optimizer.zero_grad()
        output = model(inp)
        loss = F.cross_entropy(output, target)
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


def plot(x):
    for t, v in enumerate(x):
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)
        
def plot2(x,y):
    for t, v in enumerate(x):
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)
    for t, v in enumerate(y):
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)


