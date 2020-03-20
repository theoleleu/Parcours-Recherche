from Programme import *
from torch.utils.data import Dataset
size = 1000
size2= 500
sigma = 0.6
a1=np.concatenate((np.random.normal(3, sigma, size), np.random.normal(4, sigma, size), np.random.normal(2, sigma, size),np.random.normal(5, sigma, size)))
a2=np.concatenate((np.random.normal(4, sigma, size), np.random.normal(1.6, sigma, size), np.random.normal(5, sigma, size),np.random.normal(5, sigma, size)))
b1=np.concatenate((np.random.normal(2.8, sigma, size), np.random.normal(3.7, sigma, size), np.random.normal(1.8, sigma, size),np.random.normal(5.1, sigma, size)))
b2=np.concatenate((np.random.normal(3.9, sigma, size), np.random.normal(1.8, sigma, size), np.random.normal(4.9, sigma, size),np.random.normal(5.2, sigma, size)))
c1=np.concatenate((np.random.normal(3.1, sigma, size), np.random.normal(3.7, sigma, size), np.random.normal(1.5, sigma, size),np.random.normal(4.9, sigma, size)))
c2=np.concatenate((np.random.normal(3.9, sigma, size), np.random.normal(1.5, sigma, size), np.random.normal(5.3, sigma, size),np.random.normal(4.7, sigma, size)))
at1=np.concatenate((np.random.normal(3, sigma, size), np.random.normal(4, sigma, size), np.random.normal(2, sigma, size),np.random.normal(5, sigma, size)))
at2=np.concatenate((np.random.normal(4, sigma, size), np.random.normal(1.6, sigma, size), np.random.normal(5, sigma, size),np.random.normal(5, sigma, size)))
bt1=np.concatenate((np.random.normal(2.8, sigma, size), np.random.normal(3.7, sigma, size), np.random.normal(1.8, sigma, size),np.random.normal(5.1, sigma, size)))
bt2=np.concatenate((np.random.normal(3.9, sigma, size), np.random.normal(1.8, sigma, size), np.random.normal(4.9, sigma, size),np.random.normal(5.2, sigma, size)))
ct1=np.concatenate((np.random.normal(3.1, sigma, size), np.random.normal(3.7, sigma, size), np.random.normal(1.5, sigma, size),np.random.normal(4.9, sigma, size)))
ct2=np.concatenate((np.random.normal(3.9, sigma, size), np.random.normal(1.5, sigma, size), np.random.normal(5.3, sigma, size),np.random.normal(4.7, sigma, size)))


TR1=[[[i[0],i[1]]] for i in zip(a1,a2)]
TR2=[[[i[0],i[1]]] for i in zip(b1,b2)]
TR3=[[[i[0],i[1]]] for i in zip(c1,c2)]
TE1=[[[i[0],i[1]]] for i in zip(at1,at2)]
TE2=[[[i[0],i[1]]] for i in zip(bt1,bt2)]
TE3=[[[i[0],i[1]]] for i in zip(ct1,ct2)]

train_loader,dev_loader=[0,0,0],[0,0,0]
color=[0]*1000+[1]*1000+[2]*1000+[3]*1000
train_loader[0] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TR1,color)]
train_loader[1] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TR2,color)]
train_loader[2] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TR3,color)]
dev_loader[0] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TE1,color)]
dev_loader[1] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TE2,color)]
dev_loader[2] = [(torch.tensor(i[0]),torch.tensor([i[1]])) for i in zip(TE3,color)]

##TESTS
epochs = 30
lr = 1e-3
hidden_size = 5
num_task = 3
loss, dev_loss, acc = standard_process(epochs,train_loader,dev_loader,dev_loader)
loss_plot2(loss,dev_loss)