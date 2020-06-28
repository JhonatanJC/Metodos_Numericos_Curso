#! /usr/bin/python3

import numpy as np

import matplotlib.pyplot as plt # from pylab import plot,show

import warnings
warnings.filterwarnings("ignore")

def matrices_DLU(matriz):
    D=np.zeros(matriz.shape)
    L=np.zeros(matriz.shape)
    U=np.zeros(matriz.shape)
    n=matriz.shape[1]

    for j in range(n):
        D[j][j]=matriz[j][j]
   
    for j in range(n):
        for i in range(j+1,n):
            L[i][j]=matriz[i][j]
    
    for i in range(n):
        for j in range(i+1,n):
            U[i][j]=matriz[i][j]
    
    return D,L,U

def pivoteo_parcial_filas(matriz):
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    for i in range(n):
        j=np.argmax(np.abs(matriz[i,i:])) #Nos da el indice donde el elemnto de la fila es maximo pero en la fila cortada por eso luego se le suma el indice i, para saber el verdadero indice de toda la matriz. 
        k=j+i #Verdadero indice de la columna con mayor indice
        matriz[:,[i, k]] = matriz[:,[k, i]]#Intercambia columnas.
    return matriz 

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

def is_diagonally_dominant(x):
    abs_x = np.abs(x)
    return np.all( np.diag(abs_x) >= (np.sum(abs_x, axis=1) - np.diag(abs_x))  )

"""
def Jacobi_method(matriz,vector,num_iter):
    matriz,vector= pivoteo_parcial(matriz,vector) #se hace el pivoteo parcial si es necesario.
    #D,L,U=matrices_DLU(matriz)
    n=matriz.shape[1]
    solucion=np.zeros(n)
    k=0
    while k<num_iter:
        for i in range(n):
            suma=0
            for j in range(n):
                if j != i:
                    suma=suma+matriz[i][j]*solucion[j]
            solucion[i]=(vector[i] - suma)/matriz[i][i]
        k=k+1
    if is_diagonally_dominant(matriz):
        print("Solucion parecido a GAuss")
        return solucion
    else:
        print("ALERTA!! La matriz A no es diagonalmente dominante por filas, por lo tanto la solucion presentada es probable que sea incorrecta, intentar con otro algoritmo:")
        print("\nSolucion:")
        return solucion
"""

def Jacobi_method2(matriz,vector,num_iter_max,tolerancia):
    matriz,vector= pivoteo_parcial(matriz,vector) #se hace el pivoteo parcial si es necesario.
    #D,L,U=matrices_DLU(matriz)
    n=matriz.shape[1]
    soluciones=[np.zeros(n)]
    errores=[10]#contiene el valor inicial del error
    k=0
    while k<num_iter_max and errores[k]>tolerancia:
        
        solucion=[]
        for i in range(n):
            suma=0
            for j in range(n):
                if j != i:
                    suma=suma+matriz[i][j]*soluciones[k][j]
            a=(vector[i] - suma)/matriz[i][i]
            solucion.append(a)
        soluciones.append(solucion)
        solucion=np.array(solucion)
        errores.append(np.linalg.norm(solucion-soluciones[k])/np.linalg.norm(solucion))
        k=k+1
    if is_diagonally_dominant(matriz):
        print("Solucion por JAcobi:")
        return soluciones, errores,k
    else:
        print("ALERTA!! La matriz A no es diagonalmente dominante por filas, por lo tanto la solucion presentada es probable que sea incorrecta, intentar con otro algoritmo:")
        print("\nSolucion:")
        return soluciones, errores,k


def Gauss_Seidel_method(matriz,vector,num_iter_max,tolerancia):
    matriz = pivoteo_parcial_filas(matriz) #se hace el pivoteo parcial si es necesario.
    #D,L,U=matrices_DLU(matriz)
    n=matriz.shape[1]
    solucion=np.zeros(n) #para la actualizacion inmediata de solucion
    soluciones=[np.zeros(n)]
    errores=[10]
    k=0
    while k<num_iter_max and errores[k]>tolerancia:
        solucion1=[]
        
        for i in range(0,n):
            suma1=0
            suma2=0
            for j in range(0,i):
                suma2=suma2+matriz[i][j]*solucion[j]
            for j in range(i+1,n):
                suma1=suma1+matriz[i][j]*soluciones[k][j]

            solucion[i]=(vector[i] - suma1 - suma2)/matriz[i][i]
            solucion1.append(solucion[i])
        soluciones.append(np.array(solucion1))  
        solucion1=np.array(solucion1)
        errores.append(np.linalg.norm(solucion1-soluciones[k])/np.linalg.norm(solucion1))
        k=k+1
    if is_diagonally_dominant(matriz):
        print("Solucion por GAuss Seidel:")
        return soluciones, errores,k
    else:
        print("ALERTA!! La matriz A no es diagonalmente dominante por filas, por lo tanto la solucion presentada es probable que sea incorrecta, intentar con otro algoritmo:")
        print("\nSolucion:")
        return soluciones, errores,k

def SOR(matriz,vector,num_iter_max,tolerancia,w):
    matriz = pivoteo_parcial_filas(matriz) #se hace el pivoteo parcial si es necesario.
    #D,L,U=matrices_DLU(matriz)
    n=matriz.shape[1]
    solucion=np.zeros(n) #para la actualizacion inmediata de solucion
    soluciones=[np.zeros(n)]
    errores=[10]
    k=0
    while k<num_iter_max and errores[k]>tolerancia:
        solucion1=[]
        
        for i in range(0,n):
            suma1=0
            suma2=0
            suma3=0
            for j in range(0,i):
                suma3=suma3+matriz[i][j]*solucion[j]
            for j in range(i+1,n):
                suma2=suma2+matriz[i][j]*soluciones[k][j]
            for j in range(0,i):
                suma1=suma1+matriz[i][j]*soluciones[k][j]
            solucion[i]=(vector[i] - (1-w)*suma1 - suma2 - w*suma3)/matriz[i][i]
            solucion1.append(solucion[i])
        soluciones.append(np.array(solucion1))  
        solucion1=np.array(solucion1)
        errores.append(np.linalg.norm(solucion1-soluciones[k])/np.linalg.norm(solucion1))
        k=k+1
    if is_diagonally_dominant(matriz):
        print("Solucion por SOR: con w=",w)
        return soluciones, errores,k
    else:
        print("ALERTA!! La matriz A no es diagonalmente dominante por filas, por lo tanto la solucion presentada es probable que sea incorrecta, intentar con otro algoritmo:")
        print("\nSolucion:")
        return soluciones, errores,k

matriz_prueba1=np.array([[10,1,3,-4,1],[2,-15,4,1,-1],[-1,3,-15,-1,2],[3,-2,-2,-150,3],[-4,-1,-5,3,-14]])
vector_prueba1=np.array([33,7,1,24,-49])

matriz_prueba2=np.array([[3,-0.1,-0.2],[0.1,7,-0.3],[0.3,-0.2,10]])
vector_prueba2=np.array([7.85,-19.3,71.4])

matriz_prueba3=np.array([[4,1,2,-1],[3,6,-1,2],[2,-1,5,-3],[4,1,-3,-8]])
vector_prueba3=np.array([2,-1,3,2])

matriz_prueba4=np.array([[1,5,1],[2,1,6],[8,3,2]])
vector_prueba4=np.array([7,9,13])

matriz_prueba5=np.array([[10,3,1,4,1,0],[3,-10,1,-1,2,-1],[1,3,10,-1,-2,1],[-3,-1,1,10,2,-2],[1,1,1,-2,-10,3],[2,-1,0,3,1,10]])
vector_prueba5=np.array([1,1,1,1,1,1])

matriz_prueba6=np.array([[1,1,1],[1,-1,2],[1,-1,-3]])
vector_prueba6=np.array([6,5,-10])


#print(Gauss_Seidel_method(matriz_prueba1,vector_prueba1,5))

#print(Jacobi_method(matriz_prueba2,vector_prueba2,5))
print("")

print(Jacobi_method2(matriz_prueba5,vector_prueba5,100,1e-3))
print("")

print(Gauss_Seidel_method(matriz_prueba5,vector_prueba5,100,1e-3))
print("")
print(SOR(matriz_prueba5,vector_prueba5,100,1e-3,3.0/4.0))
print("")
print(descompo_LU(matriz_prueba2,vector_prueba2))
print("")
"""
print(Jacobi_method(matriz_prueba2,vector_prueba2,200))
print(descompo_LU(matriz_prueba2,vector_prueba2))
print("")

print(Jacobi_method(matriz_prueba3,vector_prueba3,200))
print(descompo_LU(matriz_prueba3,vector_prueba3))
print("")

print(Jacobi_method(matriz_prueba4,vector_prueba4,200))
print(descompo_LU(matriz_prueba4,vector_prueba4))
print("")

print(Jacobi_method(matriz_prueba5,vector_prueba5,200))
print(descompo_LU(matriz_prueba5,vector_prueba5))
print("")

print(Jacobi_method(matriz_prueba6,vector_prueba6,200))
print(descompo_LU(matriz_prueba6,vector_prueba6))
"""




