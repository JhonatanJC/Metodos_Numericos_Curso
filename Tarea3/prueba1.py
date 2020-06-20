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

def pivoteo_parcial(matriz,vector):
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    for j in range(n):
        if matriz[j][j]==0:
            i=np.argmax(np.abs(matriz[:,j])) #Nos da el indice donde el elemnto de la columna es maximo
            matriz[[i,j]]=matriz[[j,i]]
            vector[j],vector[i]=vector[i],vector[j]
    return matriz,vector

def descompo_LU(matriz):
    #Este algoritmo esta fuertemente basado en la descompisicon gausina para backward
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    L=np.zeros([n,n])
    for j in range(n):
        L[j][j] = 1.0   #https://youtu.be/WMrwMKPhrMc (video donde se explica mejor la descomposicion LU)
        for i in range(j+1,n):
            L[i][j] = matriz[i][j]/matriz[j][j] #lA MATRIZ RESTANTE "matriz" alfinal es la matriz U buscada.
            matriz[i] = matriz[i] - L[i][j]*matriz[j]
    return L,matriz

def GAUSJORDAN(matriz,vector):
    a=Gaus_elimination_to_forward(matriz,vector)
    b=Gaus_elimination_to_backward(a[0],a[1])
    return b
#Matrices de prueba

matriz1=np.array([[2,0,0],[1,4,0],[4,3,3]])
vector1=np.array([4,2,5])

matriz2=np.array([[1,2,1],[0,-4,1],[0,0,-2]])
vector2=np.array([5,2,4])

matriz4=np.array([[2,-1,4,1,-1],[-1,3,-2,-1,2],[5,1,3,-4,1],[3,-2,-2,-2,3],[-4,-1,-5,3,-4]])
vector4=np.array([7,1,33,24,-49])

matriz3=np.array([[1,2,-6,4,5],[5,-4,1,4,6],[5,2,0,6,7],[2,3,4,5,7],[2,45,5,7,8]])
vector3=np.array([3,2,7,6,7])

print("\nProbando forward\n")
print(forward(matriz1,vector1))
print("")
print("\nProbando backward\n")
print(backward(matriz2,vector2))
print("\nPprobando eliminacion gausiana\n")
print(matriz4)
print(vector4)

print(Gaus_elimination_to_backward(matriz4,vector4))
print(Gaus_elimination_to_forward(matriz4,vector4)[0])
print("\nSOlucion con Gaus_elimination_to_backward y Gaus_elimination_to_forward\n")
print(backward(Gaus_elimination_to_backward(matriz4,vector4)[0],Gaus_elimination_to_backward(matriz4,vector4)[1]))

print(forward(Gaus_elimination_to_forward(matriz4,vector4)[0],Gaus_elimination_to_forward(matriz4,vector4)[1]))
print("\nProbando pivoteo\n")
print(matriz3,vector3)
print(pivoteo_parcial(matriz3,vector3))
print("\nDescomposicion LU\n")
print("MAtriz4\n",matriz4)
print("L\n",descompo_LU(matriz4)[0])
print("U\n",descompo_LU(matriz4)[1])
print("\n%%%%%%%%%%%\n")
print("GAUSS JORDAN\n")
print(GAUSJORDAN(matriz4,vector4))
#https://www.waxworksmath.com/Authors/G_M/Kiusalaas/NMIEW_Python/Kiusalaas.html
def cholesky(x):
    if np.all(np.linalg.eigvals(x) > 0) and np.all(x.T==x):
        return 
    else:
        return "No se puede "
    
print(cholesky(matriz4))
