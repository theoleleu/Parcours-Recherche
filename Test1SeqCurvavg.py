from SeqCurvavg import *
import numpy.random.normal as normal
import torch.tensor as tensor
import np.concatenate as concatenate

def plot(x):
    for t, v in enumerate(x):
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)
def plot2(x,y):
    for t,v in enumerate(x):
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v,label="train loss")
    for t,v in enumerate(y):
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v,label="dev loss")



from torch.utils.data import Dataset
size = 1000
sigma = 0.6
a1=concatenate((normal(3, sigma, size), normal(4, sigma, size), normal(2, sigma, size),normal(5, sigma, size)))
a2=concatenate((normal(4, sigma, size), normal(1.6, sigma, size), normal(5, sigma, size),normal(5, sigma, size)))
b1=concatenate((normal(2.8, sigma, size), normal(3.7, sigma, size), normal(1.7, sigma, size),normal(5.1, sigma, size)))
b2=concatenate((normal(3.9, sigma, size), normal(2.1, sigma, size), normal(4.9, sigma, size),normal(5.2, sigma, size)))
c1=concatenate((normal(3.1, sigma, size), normal(3.7, sigma, size), normal(1.4, sigma, size),normal(4.9, sigma, size)))
c2=concatenate((normal(3.9, sigma, size), normal(1.5, sigma, size), normal(5.3, sigma, size),normal(4.7, sigma, size)))
t1=concatenate((normal(3.1, sigma, size), normal(3.8, sigma, size), normal(1.4, sigma, size),normal(5, sigma, size)))
t2=concatenate((normal(3.9, sigma, size), normal(1.7, sigma, size), normal(5.7, sigma, size),normal(4.8, sigma, size)))
TR1=[[[i[0],i[1]]] for i in zip(a1,a2)]
TR2=[[[i[0],i[1]]] for i in zip(b1,b2)]
TR3=[[[i[0],i[1]]] for i in zip(c1,c2)]
TE1=[[[i[0],i[1]]] for i in zip(t1,t2)]

color=[0]*1000+[1]*1000+[2]*1000+[3]*1000
train_loader = [0,0,0]
test_loader=[0,0,0]
train_loader[0] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TR1,color)]
train_loader[1] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TR2,color)]
train_loader[2] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TR3,color)]
test_loader[0] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TE1,color)]
test_loader[1] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TE1,color)]
test_loader[2] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TE1,color)]


epochs = 50
lr = 1e-3
batch_size = 100
sample_size = 100
hidden_size = 80
num_task = 3

loss, dev_loss, acc = process(epochs,train_loader, test_loader,test_loader,1)