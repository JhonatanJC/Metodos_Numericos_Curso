#! /usr/bin/python3

import numpy as np

import matplotlib.pyplot as plt # from pylab import plot,show

import warnings
warnings.filterwarnings("ignore")

"""
Para más información del tema: 
-https://www.investigacionyciencia.es/blogs/matematicas/33/posts/bifurcaciones-12410
-https://youtu.be/EOvLhZPevm0
"""

#PRIMERA GRAFICA X_0(población final) vs iteracion
"""
x_0=[0.45,0.5,0.55]
r=1#2.6,3
for i in x_0:
    x=[]
    y=[]
    for j in range(1000):
        i=r*i*(1-i)
        x.append(j)
        y.append(i)
    x=np.array(x)
    y=np.array(y)
    #plt.scatter(x,y,marker="o",s=10)#, c="b")
    plt.plot(x, y, '-o')
    plt.ylabel('x_0')
    plt.xlabel('Iteración')
    plt.grid(True)
plt.show()
"""

#SEGUNDA GRAFICA  x_0(población final luego de 1000 iteraciones) vs r
x1=np.arange(1,4+0.1,0.01)#avanzando de 0.01 en 0.01 
for i in x1:
    x_00=0.5
    y=[]
    for j in range(3000):
        x_00=i*x_00*(1-x_00)
        if j>=1000:
            y.append(x_00)
        else:
            continue
    #x.append(i)
    #x=np.array(i)
    x=np.zeros(len(y))+i
    y=np.array(y)
    plt.scatter(x,y,marker="o",s=0.1)#, c='b')
    plt.ylabel('x_0, despues de mil iteraciones')
    plt.xlabel('r')
    plt.grid(True)
plt.show()

