import matplotlib.pyplot as plt
from random import *
def plotj(x,j):
    i=0
    epochs=len(x[j])
    plt.plot(list(range(i, i+epochs)), x[j], color=(random(),random(),random()))
    plt.show()
