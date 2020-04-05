from MTaskSGD import *
from numpy.random import normal
import torch.tensor as tensor
from numpy import concatenate

size = 1000
size2= 500
sigma = 0.6
a1=concatenate((normal(3, sigma, size), normal(4, sigma, size), normal(2, sigma, size),normal(5, sigma, size)))
a2=concatenate((normal(4, sigma, size), normal(1.6, sigma, size), normal(5, sigma, size),normal(5, sigma, size)))
b1=concatenate((normal(2.8, sigma, size), normal(3.7, sigma, size), normal(1.8, sigma, size),normal(5.1, sigma, size)))
b2=concatenate((normal(3.9, sigma, size), normal(1.8, sigma, size), normal(4.9, sigma, size),normal(5.2, sigma, size)))
c1=concatenate((normal(3.1, sigma, size), normal(3.7, sigma, size), normal(1.5, sigma, size),normal(4.9, sigma, size)))
c2=concatenate((normal(3.9, sigma, size), normal(1.5, sigma, size), normal(5.3, sigma, size),normal(4.7, sigma, size)))
at1=concatenate((normal(3, sigma, size), normal(4, sigma, size), normal(2, sigma, size),normal(5, sigma, size)))
at2=concatenate((normal(4, sigma, size), normal(1.6, sigma, size), normal(5, sigma, size),normal(5, sigma, size)))
bt1=concatenate((normal(2.8, sigma, size), normal(3.7, sigma, size), normal(1.8, sigma, size),normal(5.1, sigma, size)))
bt2=concatenate((normal(3.9, sigma, size), normal(1.8, sigma, size), normal(4.9, sigma, size),normal(5.2, sigma, size)))
ct1=concatenate((normal(3.1, sigma, size), normal(3.7, sigma, size), normal(1.5, sigma, size),normal(4.9, sigma, size)))
ct2=concatenate((normal(3.9, sigma, size), normal(1.5, sigma, size), normal(5.3, sigma, size),normal(4.7, sigma, size)))


TR1=[[[i[0],i[1]]] for i in zip(a1,a2)]
TR2=[[[i[0],i[1]]] for i in zip(b1,b2)]
TR3=[[[i[0],i[1]]] for i in zip(c1,c2)]
TE1=[[[i[0],i[1]]] for i in zip(at1,at2)]
TE2=[[[i[0],i[1]]] for i in zip(bt1,bt2)]
TE3=[[[i[0],i[1]]] for i in zip(ct1,ct2)]

train_loader,dev_loader=[0,0,0],[0,0,0]
color=[0]*1000+[1]*1000+[2]*1000+[3]*1000
train_loader[0] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TR1,color)]
train_loader[1] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TR2,color)]
train_loader[2] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TR3,color)]
dev_loader[0] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TE1,color)]
dev_loader[1] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TE2,color)]
dev_loader[2] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TE3,color)]

##TESTS
epochs = 30
lr = 1e-3
hidden_size = 5
num_task = 3
loss, dev_loss, acc = standard_process(epochs,train_loader,dev_loader,dev_loader)
plot2(loss,dev_loss)