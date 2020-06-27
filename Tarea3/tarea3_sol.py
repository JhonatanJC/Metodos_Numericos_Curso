#! /usr/bin/python3

import numpy as np

import matplotlib.pyplot as plt #from pylab import plot,show

import warnings
warnings.filterwarnings("ignore")

def forward(matriz,vector):
    n=matriz.shape[0] # numero de variables y tambien de ecuaciones
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    solucion=np.zeros(n)# se van guardando los valores de las soluciones encontradas para luego ser utilizadas. inicialmente en 0 .
    for j in range(0,n):
        suma=0 
        for i in range(0,n-1): # PARA LA SUMA se empieza de 0 porque vamos a encontrar la solucion hacia "adelante"
            suma=suma + matriz[j][i]*solucion[i]    #suma que utiliza las soluciones ya encontradas
        solucion[j] = (vector[j] - suma)/matriz[j][j] # formula de recurencia     
    return solucion

def backward(matriz,vector):
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    solucion=np.zeros(n) 
    for j in range(n-1,-1,-1): #Se comienza desde n-1 porque el indice de una array empieza de 0 a n-1 y se va aretroceder en este caso . donde n es el tamaño de la dimension, j toma los valores {2,1,0} , se comenza desde el indice n-1 porque vamos a "retroceder".
        suma=0
        for i in range(j+1,n): 
            suma=suma + matriz[j][i]*solucion[i]    # suma #suma que utiliza las soluciones ya encontradas 
        solucion[j] = (vector[j] - suma)/matriz[j][j] # formula de recurencia 
    return solucion

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

def Gaus_elimination_to_backward(matriz,vector):
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    if np.trace(matriz)==0: 
        matriz=pivoteo_parcial(matriz,vector)[0]    # Se implementa el pivoteo en casa que sea necesario
        vector=pivoteo_parcial(matriz,vector)[1]
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
    if np.trace(matriz)==0: 
        matriz=pivoteo_parcial(matriz,vector)[0]    # Se implementa el pivoteo en casa que sea necesario
        vector=pivoteo_parcial(matriz,vector)[1]
    for j in range(n-1,-1,-1):
        vector[j]=vector[j]/matriz[j][j]#EN ESTA PARTE HACEMOS 1 A LA DIAGONAL PRINCIPAL
        matriz[j]=matriz[j]/matriz[j][j]
        for i in range(j-1,-1,-1):
            vector[i]=vector[i] - matriz[i][j]*vector[j] # aplicando la eliminacion de gaus, restando primera fila por las siguientes hasta hacer ceroo la primera columna menos el primer elemento, luego restando la segunda fila por las filas restantes de abajo hasta hacer ceros los numeros a partir de esa columna hacia bajo.
            matriz[i]=matriz[i] - matriz[i][j]*matriz[j]
    return matriz,vector


def Gauss_Jordan(matriz,vector):
    if np.trace(matriz)==0: 
        matriz=pivoteo_parcial(matriz,vector)[0]    # Se implementa el pivoteo en casa que sea necesario
        vector=pivoteo_parcial(matriz,vector)[1]
    a=Gaus_elimination_to_forward(matriz,vector)
    b=Gaus_elimination_to_backward(a[0],a[1])
    solucion=b[1]
    return solucion

def descompo_LU(matriz,vector):
    #Este algoritmo esta fuertemente basado en la descompisicion gausina para backward
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    L=np.zeros([n,n])
    for j in range(n):
        L[j][j] = 1.0   #https://youtu.be/WMrwMKPhrMc (video donde se explica mejor la descomposicion LU)
        for i in range(j+1,n):
            L[i][j] = matriz[i][j]/matriz[j][j] #lA MATRIZ RESTANTE "matriz" alfinal es la matriz U buscada.
            matriz[i] = matriz[i] - L[i][j]*matriz[j]
    U=matriz
    b=forward(L,vector)     #Se aplican los pasos finales para dar con la solucion
    solucion=backward(U,b)
    return solucion

def descompo_LU2(matriz,vector): #APLICANDO EL ALGORITMO DE CROUT
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    L=np.zeros([n,n])
    U=np.zeros([n,n])
    for j in range(n):
        L[j][j] = 1.0
        for i in range(0,j+1):
            suma=0
            for k in range(0,i):
                suma =suma + L[i][k]*U[k][j]
            U[i][j]=matriz[i][j] - suma
        
        for i in range(0,j):    #Este bucle hace ceros la parte superior porque es una matriz triangular superior.
            U[j][i]=0

        for i in range(j,n):
            suma=0
            for k in range(0,j):
                suma=suma + L[i][k]*U[k][j]
            L[i][j]=(matriz[i][j] - suma)/U[j][j]
            
        for i in range(0,j):    #Este bucle hace ceros la parte superior porque es una matriz triangular inferior.
            L[i][j]=0

    b=forward(L,vector)     #Se aplican los pasos finales para dar con la solucion
    solucion=backward(U,b)   
    return solucion


def descomp_cholesky(matriz,vector):
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    if np.all(np.linalg.eigvals(matriz) > 0) and np.all(matriz.T==matriz):
        L_cho=np.zeros([n,n])
        for j in range(n):
            suma=0
            for p in range(0,j): # Se deben ejecutar n-1 iteraciones según la formula
                suma =suma+(L_cho[j][p])**2
            L_cho[j][j]=np.sqrt(matriz[j][j] - suma)
            for i in range(n):
                suma=0
                for p in range(0,i):
                    suma=suma + L_cho[j][p]*L_cho[i][p]
                L_cho[i][j]=(matriz[i][j] - suma)/L_cho[j][j]
            
            for i in range(0,j):    #Este bucle hace ceros la parte superior porque es una matriz triangular inferior.
                L_cho[i][j]=0
        U_cho=L_cho.T
        b=forward(L_cho,vector)     #Se aplican los pasos finales para dar con la solucion
        solucion=backward(U_cho,b)   
        return solucion
    else:
        return "No se puede hacer la descomposicion Cholesky."

print("\n###EJERCICIOS###:\n")
print("EJERCICIO1\n")
matriz1=np.array([[2,0,0],[1,4,0],[4,3,3]])
vector1=np.array([4,2,5])

matriz2=np.array([[1,2,1],[0,-4,1],[0,0,-2]])
vector2=np.array([5,2,4])
print("\n###################################################################")
print("Probando Sustitución directa(forward)\n")
print("Matriz\n",matriz1)
print("\nVector\n",vector1)
print("\nSOLUCION:",forward(matriz1,vector1))

print("\n###################################################################")
print("Probando Sustitución inversa(backward)\n")
print("Matriz\n",matriz2)
print("\nVector\n",vector2)
print("\nSOLUCION:",backward(matriz2,vector2))

print("EJERCICIO2\n")
"""
def matriz_probl2(n):
    matriz=np.zeros([n,n])
    for j in range(n):
        if j==0:
            for i in range(j,j+2):
                matriz[i][j]=-1
        elif j>0 and j<n-1:
            for i in range(j-1,j+2):
                matriz[i][j]=-1
        else:
            for i in range(j-1,j+1):
                matriz[i][j]=-1
        matriz[j][j]=3.0
    return matriz
n=3
matriz_prob_2=matriz_probl2(n)
vector_prob_2=np.zeros(n)
print("MAtriz\n",matriz_prob_2)
print("Vector\n",vector_prob_2)
print("\nSOLUCION POR MEDIO DE LOS DIFERENTES ALGORITMOS Y METODOS:")
print("\nGAUSS-JORDAN\n")
print(Gauss_Jordan(matriz_prob_2,vector_prob_2))
print("\nDESCOMPOSICION LU(PRIMER METODO)\n")
#print(descompo_LU(matriz_prob_2,vector_prob_2))
print("\nDESCOMPOSICION LU(ALGORITMO DE CROUT)\n")
#print(descompo_LU2(matriz_prob_2,vector_prob_2))
print("\nDESCOMPOSICION CHOLESKY\n")
#print(descomp_cholesky(matriz_prob_2,vector_prob_2))
"""

print("EJERCICIO3\n")
k=10
g=9.8
m1=m2=m3=1
matriz3=np.array([[3*k,-2*k,0],[-2*k,3*k,-k],[0,-k,k]])
vector3=np.array([m1*g,m2*g,m3*g])
print("MAtriz\n",matriz3)
print("Vector\n",vector3)
print("\nSOLUCION POR MEDIO DE LOS DIFERENTES ALGORITMOS Y METODOS:")
print("\nGAUSS-JORDAN\n")
print(Gauss_Jordan(matriz3,vector3))
print("\nDESCOMPOSICION LU(PRIMER METODO)\n")
print(descompo_LU(matriz3,vector3))
print("\nDESCOMPOSICION LU(ALGORITMO DE CROUT)\n")
print(descompo_LU2(matriz3,vector3))
print("\nDESCOMPOSICION CHOLESKY\n")
print(descomp_cholesky(matriz3,vector3))
