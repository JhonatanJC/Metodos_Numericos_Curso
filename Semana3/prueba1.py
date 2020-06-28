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

def Gauss_Jordan(matriz,vector):
    a=Gaus_elimination_to_forward(matriz,vector)
    b=Gaus_elimination_to_backward(a[0],a[1])
    return b

def descompo_LU(matriz):
    #Este algoritmo esta fuertemente basado en la descompisicion gausina para backward
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    L=np.zeros([n,n])
    for j in range(n):
        L[j][j] = 1.0   #https://youtu.be/WMrwMKPhrMc (video donde se explica mejor la descomposicion LU)
        for i in range(j+1,n):
            L[i][j] = matriz[i][j]/matriz[j][j] #lA MATRIZ RESTANTE "matriz" alfinal es la matriz U buscada.
            matriz[i] = matriz[i] - L[i][j]*matriz[j]
    U=matriz
    return L,U

def descompo_LU2(matriz): #APLICANDO EL ALGORITMO DE CROUT
    n=matriz.shape[0]
    matriz=matriz.astype(float)
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
    
    return L, U, L@U


def descomp_cholesky(x):
    if np.all(np.linalg.eigvals(x) > 0) and np.all(x.T==x):
        n=x.shape[0]
        L_cho=np.zeros([n,n])
        for j in range(n):
            suma=0
            for p in range(0,j): # Se deben ejecutar n-1 iteraciones según la formula
                suma =suma+(L_cho[j][p])**2
            L_cho[j][j]=np.sqrt(x[j][j] - suma)
            for i in range(n):
                suma=0
                for p in range(0,i):
                    suma=suma + L_cho[j][p]*L_cho[i][p]
                L_cho[i][j]=(x[i][j] - suma)/L_cho[j][j]
            
            for i in range(0,j):    #Este bucle hace ceros la parte superior porque es una matriz triangular inferior.
                L_cho[i][j]=0
        U_cho=L_cho.T
        return L_cho,U_cho,L_cho@U_cho
    else:
        return "No se puede hacer la descomposicion Cholesky."

#Matrices de prueba
matriz1=np.array([[2,0,0],[1,4,0],[4,3,3]])
vector1=np.array([4,2,5])

matriz2=np.array([[1,2,1],[0,-4,1],[0,0,-2]])
vector2=np.array([5,2,4])

matriz_piv=np.array([[1,2,-6,4,5],[5,-4,1,4,6],[5,2,8,6,7],[2,3,4,0,7],[2,45,5,7,8]])
vector_piv=np.array([3,2,7,6,7])

matriz4=np.array([[2,-1,4,1,-1],[-1,3,-2,-1,2],[5,1,3,-4,1],[3,-2,-2,-2,3],[-4,-1,-5,3,-4]])
vector4=np.array([7,1,33,24,-49])

matriz5=np.array([[6,15,55],[15,55,225],[55,225,979]])
vector5=np.array([100,150,100])

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

print("\n###################################################################")
print("Probando Algoritmo de eliminacion Gausiana\n")
print("Matriz\n",matriz4)
print("\nVector\n",vector4)

print("\nMatriz y vector despues de aplicar Eliminación Gausiana para aplicar FORWARD:\n")
print(Gaus_elimination_to_forward(matriz4,vector4)[0])
print("")
print(Gaus_elimination_to_forward(matriz4,vector4)[1])
print("\nSOLUCION PARA FORWARD:",forward(Gaus_elimination_to_forward(matriz4,vector4)[0],Gaus_elimination_to_forward(matriz4,vector4)[1]) )

print("\nMatriz y vector despues de aplicar Eliminación Gausiana para aplicar BACKWARD:\n")
print(Gaus_elimination_to_backward(matriz4,vector4)[0])
print("")
print(Gaus_elimination_to_backward(matriz4,vector4)[1])
print("\nSOLUCION PARA BACKWARD:",backward(Gaus_elimination_to_backward(matriz4,vector4)[0],Gaus_elimination_to_backward(matriz4,vector4)[1]))

print("\n###################################################################")
print("Probando Algoritmo de Pivoteo parcial\n")

print("Matriz\n",matriz_piv)
print("\nVector\n",vector_piv)
print("\nDespues del pivoteo")
print("\nMatriz")
print(pivoteo_parcial(matriz_piv,vector_piv)[0])
print("\nVector")
print(pivoteo_parcial(matriz_piv,vector_piv)[1])

print("\n###################################################################")
print("Probando Algoritmo de Gauss-Jordan\n")
print("Matriz\n",matriz4)
print("\nVector\n",vector4)

print("\nDespues de aplicar el algoritmo de Eliminación Gausiana para BACKWARD y Eliminación Gausiana para FORWARD de manera consecutiva (GAuss-Jordan)\n")
print("Matriz\n",Gauss_Jordan(matriz4,vector4)[0])
print("Vector\n",Gauss_Jordan(matriz4,vector4)[1])
print("\nSOLUCION:",Gauss_Jordan(matriz4,vector4)[1])

print("\n###################################################################")
print("Probando Algoritmo de descomposicion LU(tutorial)\n")
print("Matriz\n",matriz4)
print("\nSE OBTIENE LAS SIGUIENTES MATRICES\n")
print("Matriz L\n",descompo_LU(matriz4)[0])
print("\nMAtriz U\n",descompo_LU(matriz4)[1])
print("Comprobando que el productor de A=L*U\n",descompo_LU2(matriz4)[2])

print("\n###################################################################")
print("Probando Algoritmo de descomposicion LU(algoritmo CROUT)\n")
print("Matriz\n",matriz4)
print("\nSE OBTIENE LAS SIGUIENTES MATRICES\n")
print("MAtriz L\n",descompo_LU2(matriz4)[0])
print("\nMatriz U\n",descompo_LU2(matriz4)[1])
print("\nComprobando que el productor de A=L*U\n",descompo_LU2(matriz4)[2])

print("\n###################################################################")
print("Probando Algoritmo de descomposicion Cholesky(tutorial) para la matriz antes usada\n")
print("Matriz\n",matriz4)
print("\nVector\n",vector4)
print("\nSe debe comprobar que cumplan las concidiciones para que se puede hacer la descomposicion CHolesky:\n")
print(descomp_cholesky(matriz4))

print("\n###################################################################")
print("Probando Algoritmo de descomposicion Cholesky(tutorial) para otra matriz\n")

print("Matriz\n",matriz5)
print("\nSE OBTIENE LAS SIGUIENTES MATRICES\n")
print("MAtriz L\n",descomp_cholesky(matriz5)[0])
print("\nMatriz U=L(tranpuesta)\n",descomp_cholesky(matriz5)[1])
print("\nComprobando que el productor de A=L*U\n",descomp_cholesky(matriz5)[2])
