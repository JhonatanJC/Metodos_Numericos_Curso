#! /usr/bin/python3

import numpy as np

import matplotlib.pyplot as plt #from pylab import plot,show

import warnings
warnings.filterwarnings("ignore")


def forward(matriz,vector):
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    solucion=np.zeros(len(vector))# se van guardando los valores de las soluciones encontradas para luego ser utilizadas. inicialmente en 0 .
    n=matriz.shape[0] # numero de variables y tambien de ecuaciones
    for j in range(0,n):
        suma=0 
        for i in range(0,n-1): # PARA LA SUMA se empieza de 0 porque vamos a encontrar la solucion hacia "adelante"
            suma=suma + matriz[j][i]*solucion[i]    #suma que utiliza las soluciones ya encontradas
        solucion[j] = (vector[j] - suma)/matriz[j][j] # formula de recurencia     
    return solucion

def backward(matriz,vector):
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    solucion=np.zeros(len(vector))
    n=matriz.shape[0]
    for j in range(n-1,-1,-1): #Se comienza desde n-1 porque el indice de una array empieza de 0 a n-1 y se va aretroceder en este caso . donde n es el tamaño de la dimension, j toma los valores {2,1,0} , se comenza desde el indice n-1 porque vamos a "retroceder".
        suma=0
        for i in range(j+1,n): 
            suma=suma + matriz[j][i]*solucion[i]    # suma #suma que utiliza las soluciones ya encontradas 
        solucion[j] = (vector[j] - suma)/matriz[j][j] # formula de recurencia 
    return solucion

def Gaus_elimination_to_backward(matriz,vector):
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    for j in range(0,n):
        vector[j]=vector[j]/matriz[j][j]#EN ESTA PARTE HACEMOS 1 A LA DIAGONAL PRINCIPAL
        matriz[j]=matriz[j]/matriz[j][j]
        for i in range(j+1,n):
            vector[i]=vector[i] - matriz[i][j]*vector[j] # aplicando la eliminacion de gaus, restando primera fila por las siguientes hasta hacer ceroo la primera columna menos el primer elemento, luego restando la segunda fila por las filas restantes de abajo hasta hacer ceros los numeros a partir de esa columna hacia bajo.
            matriz[i]=matriz[i] - matriz[i][j]*matriz[j]
    return matriz,vector


def Gaus_elimination_to_forward(matriz,vector):
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    for j in range(n-1,-1,-1):
        vector[j]=vector[j]/matriz[j][j]#EN ESTA PARTE HACEMOS 1 A LA DIAGONAL PRINCIPAL
        matriz[j]=matriz[j]/matriz[j][j]
        for i in range(j-1,-1,-1):
            vector[i]=vector[i] - matriz[i][j]*vector[j] # aplicando la eliminacion de gaus, restando primera fila por las siguientes hasta hacer ceroo la primera columna menos el primer elemento, luego restando la segunda fila por las filas restantes de abajo hasta hacer ceros los numeros a partir de esa columna hacia bajo.
            matriz[i]=matriz[i] - matriz[i][j]*matriz[j]
    return matriz,vector

#Matrices de prueba
matriz1=np.array([[2,0,0],[1,4,0],[4,3,3]])
vector1=np.array([4,2,5])

matriz2=np.array([[1,2,1],[0,-4,1],[0,0,-2]])
vector2=np.array([5,2,4])

matriz3=np.array([[5,2,1,4,5],[5,-4,1,4,6],[3,2,-2,6,7],[2,3,4,5,7],[2,45,6,7,8]])
vector3=np.array([3,2,7,6,7])

matriz4=np.array([[2,-1,4,1,-1],[-1,3,-2,-1,2],[5,1,3,-4,1],[3,-2,-2,-2,3],[-4,-1,-5,3,-4]])
vector4=np.array([7,1,33,24,-49])

print(forward(matriz1,vector1))
print("")
print(backward(matriz2,vector2))
print(matriz4)
print(vector4)
print(Gaus_elimination_to_backward(matriz4,vector4))
print(Gaus_elimination_to_forward(matriz4,vector4)[0])

print(backward(Gaus_elimination_to_backward(matriz4,vector4)[0],Gaus_elimination_to_backward(matriz4,vector4)[1]))

print(forward(Gaus_elimination_to_forward(matriz4,vector4)[0],Gaus_elimination_to_forward(matriz4,vector4)[1]))


