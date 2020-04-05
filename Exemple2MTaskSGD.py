from MTaskSGD import *
from numpy.random import normal
import torch.tensor as tensor
from numpy import concatenate

size = 1000
size2= 500
sigma = 0.6
a1=concatenate((normal(2, sigma, size), normal(4, sigma, size), normal(1, sigma, size),normal(5, sigma, size)))#Le paramètre de la distribution réalise des classes
a2 = concatenate((normal(4, sigma, size), normal(1.8, sigma, size), normal(7, sigma, size),normal(5, sigma, size)))
b1= concatenate((normal(2.8, sigma, size), normal(3.7, sigma, size), normal(1.7, sigma, size),normal(5.4, sigma, size)))#Le paramètre de la distribution réalise des classes
b2 = concatenate((normal(3.9, sigma, size), normal(1.5, sigma, size), normal(6.4, sigma, size),normal(5.2, sigma, size)))
c1=concatenate((normal(3.1, sigma, size), normal(3.7, sigma, size), normal(1.4, sigma, size),normal(5.7, sigma, size)))#Le paramètre de la distribution réalise des classes
c2 = concatenate((normal(3.9, sigma, size), normal(1.5, sigma, size), normal(6, sigma, size),normal(4.7, sigma, size)))
t1=concatenate((normal(3.1, sigma, size2), normal(3.8, sigma, size2), normal(1.4, sigma, size2),normal(5.4, sigma, size2)))#Le paramètre de la distribution réalise des classes
t2 = concatenate((normal(3.9, sigma, size2), normal(1.4, sigma, size2), normal(5.8, sigma, size2),normal(4.8, sigma, size2)))
t3 = concatenate((normal(3.1, sigma, size2), normal(3.8, sigma, size2), normal(1.4, sigma, size2),normal(5, sigma, size2)))#Le paramètre de la distribution réalise des classes
t4 = concatenate((normal(3.9, sigma, size2), normal(1.7, sigma, size2), normal(5.7, sigma, size2),normal(4.8, sigma, size2)))
t5=concatenate((normal(3.1, sigma, size2), normal(3.8, sigma, size2), normal(1.4, sigma, size2),normal(5.4, sigma, size2)))#Le paramètre de la distribution réalise des classes
t6 = concatenate((normal(3.9, sigma, size2), normal(1.7, sigma, size2), normal(5.7, sigma, size2),normal(4.8, sigma, size2)))
at1=concatenate((normal(2, sigma, size2), normal(3.9, sigma, size2), normal(1, sigma, size2),normal(5, sigma, size2)))#Le paramètre de la distribution réalise des classes
at2 = concatenate((normal(4, sigma, size2), normal(1.8, sigma, size2), normal(6.95, sigma, size2),normal(5, sigma, size2)))
TR1=[[[i[0],i[1]]] for i in zip(a1,a2)]
TR2=[[[i[0],i[1]]] for i in zip(b1,b2)]
TR3=[[[i[0],i[1]]] for i in zip(c1,c2)]
TE1=[[[i[0],i[1]]] for i in zip(t1,t2)]
TE2=[[[i[0],i[1]]] for i in zip(t3,t4)]
TE3=[[[i[0],i[1]]] for i in zip(t5,t6)]
TT1=[[[i[0],i[1]]] for i in zip(at1,at2)]
train_loader=[0,0,0]
test_loader=[0,0,0]
dev_loader=[0,0,0]
color2=[0]*size2+[1]*size2+[2]*size2+[3]*size2
color=[0]*1000+[1]*size+[2]*size+[3]*size
train_loader[0] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TR1,color)]
train_loader[1] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TR2,color)]
train_loader[2] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TR3,color)]
test_loader[0] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TT1,color2)]
test_loader[1] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TE2,color2)]
test_loader[2] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TE3,color2)]
dev_loader[0] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TE1,color2)]
dev_loader[1] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TE2,color2)]
dev_loader[2] = [(tensor(i[0]),tensor([i[1]])) for i in zip(TE3,color2)]

#Test
epochs = 50
lr = 1e-3
batch_size = 100
sample_size = 100
hidden_size = 80
num_task = 3

loss, dev_loss, acc = process(epochs,train_loader,dev_loader,test_loader)
plot2(loss,dev_loss)
plot(acc)