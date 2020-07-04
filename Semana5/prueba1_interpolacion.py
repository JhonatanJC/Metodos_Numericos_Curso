#! /usr/bin/python3

import numpy as np

import matplotlib.pyplot as plt # from pylab import plot,show

import warnings
warnings.filterwarnings("ignore")

from numpy import genfromtxt

#my_data = genfromtxt('my_file.csv', delimiter=',')

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

########################################################
########################################################
##METODO DIRECTO PARA RESOLVER UN SISTEMA DE ECUACIONES LINEALES

def forward(matriz,vector):
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    solucion=np.zeros(n)
    for j in range(0,n):
        suma=0 
        for i in range(0,n-1): 
            suma=suma + matriz[j][i]*solucion[i]   
        solucion[j] = (vector[j] - suma)/matriz[j][j]     
    return solucion
def backward(matriz,vector):
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    solucion=np.zeros(n) 
    for j in range(n-1,-1,-1):
        suma=0
        for i in range(j+1,n):
            suma=suma + matriz[j][i]*solucion[i]   
        solucion[j] = (vector[j] - suma)/matriz[j][j] 
    return solucion
def descompo_LU(matriz,vector):
    matriz,vector = pivoteo_parcial(matriz,vector)
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    L=np.zeros([n,n])
    for j in range(n):
        L[j][j] = 1.0
        for i in range(j+1,n):
            L[i][j] = matriz[i][j]/matriz[j][j] 
            matriz[i] = matriz[i] - L[i][j]*matriz[j]
    U=matriz
    b=forward(L,vector)
    solucion=backward(U,b)
    return solucion
################################################################
################################################################

def is_diagonally_dominant(x):
    abs_x = np.abs(x)
    return np.all( np.diag(abs_x) >= (np.sum(abs_x, axis=1) - np.diag(abs_x))  )

def Jacobi_method(matriz,vector,num_iter_max,tolerancia):
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
        return soluciones, errores,k
    else:
        print("ALERTA!! La matriz A no es diagonalmente dominante por filas, por lo tanto la solucion presentada es probable que sea incorrecta, intentar con otro algoritmo:")
        print("\nSolucion:")
        return soluciones, errores,k

def vandermonde(vector):
    vector=np.array(vector)
    matriz=np.zeros([vector.shape[0],vector.shape[0]])
    for j in range(vector.shape[0]):
        matriz[:,j]= vector**(j)
    return matriz

def monomio(datos_x,datos_y):
    matriz=vandermonde(datos_x)
    vector=datos_y
    #solucion1= Jacobi_method(matriz,vector,100,1e-5)#por metodos iterativos
    solucion2= descompo_LU(matriz,vector)#por metodo directo
    return solucion2#,solucion1

x_prueba0=np.array([-2,0,1])
y_prueba0=np.array([-27,-1,0])
print("Monomio: \n",monomio(x_prueba0,y_prueba0))


##################################################################
def l_i(x,datos_x):
    n=datos_x.shape[0] - 1 #n=numero de punto o datos menos 1 .
    l_s=[]
    for k in range(datos_x.shape[0]):
        producto=1
        for j in range(datos_x.shape[0]):
            if k!=j:
                producto = producto*((x-datos_x[j])/(datos_x[j]-datos_x[k]))
        l_s.append(producto)
    l_s=np.array(l_s)
    return l_s

def Pol_Lagrange(x,datos_x,datos_y):
    l=l_i(x,datos_x)
    y=datos_y@l
    return y

x_prueba=np.array([0,2,3])
y_prueba=np.array([7,11,28])

print("POlinomio de LAgrange: \n",Pol_Lagrange(1,x_prueba,y_prueba))

######################################################################
#Ver Numerical Methods in Engineering with Python 3 pag 108
def Pol_Newton(x,datos_x,datos_y):
    m = datos_x.shape[0]
    n=m-1
    a = datos_y
    
    for k in range(1,m): #EN esta parte se calcula los coeficientes.
        a[k:m] = (a[k:m] - a[k-1])/(datos_x[k:m] - datos_x[k-1])    
    
    p = a[n]
    for k in range(1,n+1):  #EN esta parte se calulan los polinomios.
        p = a[n-k] + (x -datos_x[n-k])*p
    return p

x_prueba2=np.array([-2,-1,1,3])
y_prueba2=np.array([-35,-11,-5,25])

print("Polinomio de Newton: \n",Pol_Newton(-3,x_prueba2,y_prueba2))
