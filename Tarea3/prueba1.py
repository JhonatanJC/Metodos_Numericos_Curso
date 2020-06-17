#! /usr/bin/python3

import numpy as np

import matplotlib.pyplot as plt #from pylab import plot,show

import warnings
warnings.filterwarnings("ignore")

matriz1=np.array([[2,0,0],[1,4,0],[4,3,3]])
vector1=np.array([4,2,5])

matriz2=np.array([[1,2,1],[0,-4,1],[0,0,-2]])
vector2=np.array([5,2,4])

def forward(matriz,vector):
    solucion=np.zeros(len(vector))# se van guardando los valores de las soluciones encontradas para luego ser utilizadas. inicialmente en 0 .
    n=matriz.shape[0] # numero de variables y tambien de ecuaciones
    for j in range(0,n):
        suma=0 
        for i in range(0,n-1): #se empieza de 0 porque vamos a encontrar la solucion hacia "adelante"
            suma=suma + matriz[j][i]*solucion[i]    #suma que utiliza las soluciones ya encontradas
        solucion[j] = (vector[j] - suma)/matriz[j][j] # formula de recurencia     
    return solucion

def backward(matriz,vector):
    solucion=np.zeros(len(vector))
    n=matriz.shape[0]
    for j in range(n-1,-1,-1): #Se comienza desde n-1 porque el indice de una array empieza de 0 a n-1 y se va aretroceder en este caso . donde n es el tamaño de la dimension, j toma los valores {2,1,0} , se comenza desde el indice n-1 porque vamos a "retroceder".
        suma=0
        for i in range(j+1,n): 
            suma=suma + matriz[j][i]*solucion[i]    # suma #suma que utiliza las soluciones ya encontradas 
        solucion[j] = (vector[j] - suma)/matriz[j][j] # formula de recurencia 
    return solucion

print(forward(matriz1,vector1))
print("")
print(backward(matriz2,vector2))
