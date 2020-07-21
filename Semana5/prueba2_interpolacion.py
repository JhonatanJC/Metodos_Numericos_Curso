#! /usr/bin/python3

import numpy as np

import matplotlib.pyplot as plt # from pylab import plot,show

import warnings
warnings.filterwarnings("ignore")

from numpy import genfromtxt

#my_data = genfromtxt('my_file.csv', delimiter=',')

########################################################
########################################################
##METODO DIRECTO PARA RESOLVER UN SISTEMA DE ECUACIONES LINEALES

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
##########################################################################

def monomio(x,datos_x,datos_y):
    """Recibe como entrada x un vector de valores numericos"""
    matriz=np.zeros([datos_x.shape[0],datos_x.shape[0]])
    for j in range(datos_x.shape[0]): #Se contruye la matriz de vandermonde
        matriz[:,j]= datos_x**(j)
    matriz,datos_y=pivoteo_parcial(matriz,datos_y)
    x1= descompo_LU(matriz,datos_y)# se resulve el sistema de ecuaciones por metodo directo

    puntos=[] #se almacenan los valores de y  para cada punto de x que se quiera calcular 

    for p in x: #va a ir tomando los valores de x uno por uno 
        prod=np.zeros(x1.shape[0])
        for i in range(x1.shape[0]):
            if i==0:
                prod[i]=1
            else:
                prod[i]=prod[i-1]*p #Se hace el calculo de los polimonios con todos los valores de x 
        solucion=x1@prod
        puntos.append(solucion) # se agregan los valores de y a la lista final 
    puntos=np.array(puntos)# se convierte la lista en array para mejor manejo

    return puntos

##################################################################

def Pol_Lagrange(x,datos_x,datos_y):
    """Recibe como entrada x un vector de valores numericos"""
    puntos=[]
    for p in x:               
        n=datos_x.shape[0] - 1 #n=numero de punto o datos menos 1 .
        l_s=[]
        for k in range(datos_x.shape[0]):
            producto=1
            for j in range(datos_x.shape[0]):
                if k!=j:
                    producto = producto*((p-datos_x[j])/(datos_x[k]-datos_x[j]))
            l_s.append(producto)
        l_s=np.array(l_s)
        puntos.append(datos_y@l_s)
    puntos=np.array(puntos)
    return puntos

######################################################################
# https://zaguan.unizar.es/record/15161/files/TAZ-TFM-2014-343.pdf pagina 8
def Pol_Newton_un_punto(x,datos_x,datos_y):
    """Recibe como entrada x un valor numerico"""
    n = datos_x.shape[0]
    matriz=np.ones([n,n])
    for j in range(n):
        for i in range(n):
            if j>i:
                matriz[i][j]=0
            else:
                producto=1
                for k in range(j):
                    producto=producto*(datos_x[i]-datos_x[k])
                matriz[i][j]=producto
    matriz,datos_y1= pivoteo_parcial(matriz,datos_y)
    x1 = descompo_LU(matriz,datos_y1)
    prod=np.zeros(x1.shape[0])
    for i in range(n):
        if i==0:
            prod[i]=1
        else: 
            prod[i]=prod[i-1]*(x-datos_x[i-1])
    solucion=x1@prod
    return solucion
#################################################33
##PARA VARIOS PUNTOS
def Pol_Newton(x,datos_x,datos_y):
    """Recibe como entrada x un vector de valores numericos"""
    puntos=[]
    for p in x:
        n = datos_x.shape[0]
        matriz=np.ones([n,n])
        for j in range(n):
            for i in range(n):
                if j>i:
                    matriz[i][j]=0
                else:
                    producto=1
                    for k in range(j):
                        producto=producto*(datos_x[i]-datos_x[k])
                    matriz[i][j]=producto
        matriz,datos_y1= pivoteo_parcial(matriz,datos_y)
        x1 = descompo_LU(matriz,datos_y1)
        prod=np.zeros(x1.shape[0])
        for i in range(n):
            if i==0:
                prod[i]=1
            else: 
                prod[i]=prod[i-1]*(p-datos_x[i-1])
        solucion=x1@prod
        puntos.append(solucion)
    puntos=np.array(puntos)
    return puntos

######################################################################
#Ver Numerical Methods in Engineering with Python 3 pag 108
def Pol_Newton2(x,datos_x,datos_y):
    m = datos_x.shape[0]
    n=m-1
    a = np.copy(datos_y)
    
    for k in range(1,m): #EN esta parte se calcula los coeficientes.
        a[k:m] = (a[k:m] - a[k-1])/(datos_x[k:m] - datos_x[k-1])    
    
    p = a[n]
    for k in range(1,n+1):  #EN esta parte se calulan los polinomios.
        p = a[n-k] + (x -datos_x[n-k])*p
    return p
########################################################################3

x_tarea=np.array([0,1,2,5.5,11,13,16,18])
y_tarea=np.array([0.5,3.134,5.3,9.9,10.2,9.35,7.2,6.2])


#print("Monomio: \n",monomio([8],x_tarea,y_tarea))
#print("Polinomio de Lagrange: \n",Pol_Lagrange([8],x_tarea,y_tarea))
#print("Polinomio de Newton: \n",Pol_Newton([8],x_tarea,y_tarea))

##############################################################################
#################################################

#http://www-eio.upc.es/~nasini/Blog/ChebyshevPolinomio.pdf
def f(x):
    return np.tanh(20*np.sin(12*x)) + (np.exp(3*x)*np.sin(300*x))/50.0
def f1(x):
    return 1.0/(1.0+x**2)
def Chebyshev(puntos_x,a,b,n,f):
    m=n-1#Grados del polinomio
    x=[]
    for i in range(0,m+1):
        x_i=a + ((b-a)/2.0)*(1 + np.cos(i*np.pi/n))
        x.append(x_i)
    x_Cheby=np.array(x)
    y_Cheby=f(x_Cheby) #x e y son los puntos de chebyshev
    y_Lagrange=Pol_Lagrange(puntos_x, x_Cheby,y_Cheby)
    return x_Cheby, y_Cheby,y_Lagrange

puntos_x=np.linspace(0,1,1000)

y_che=Chebyshev(puntos_x,0,1,100,f)
x_Cheby, y_Cheby = y_che[0],y_che[1]

puntos_equi=np.linspace(0,1,100)
y_equid=Pol_Lagrange(puntos_x, puntos_equi,f(puntos_equi))

plt.figure()
plt.plot(puntos_x,f(puntos_x),linestyle="-",label="Funcion Real",color="green")
plt.scatter(x_Cheby,y_Cheby,label="Datos experimentales(Puntos de Chebyshev)",color="red",s=20)
plt.plot(puntos_x,y_che[2],label="Interp. Chebyshev",linestyle=':',color="blue")
plt.plot(puntos_x,y_equid,label="Interp. pts equidistantes",linestyle='--',color="black")
plt.legend(loc="best")
plt.xlabel("F(x)")
plt.ylim((-1.5,1.5))
plt.ylabel("x")
plt.title("GRAFICA DE FUNCIONES POR MEDIO DE LA INTERPOOLACION POLINOMICA")
plt.grid()
plt.show()

#GRAFICA DE ERRORES
error_chev= np.abs(f(puntos_x)-y_che[2] )
error_equid= np.abs(f(puntos_x)-y_equid )

plt.figure()
plt.plot(puntos_x,error_chev,label="Interp. Chebyshev",linestyle=':',color="blue")
plt.plot(puntos_x,error_equid,label="Interp. pts equidistantes",linestyle='--',color="black")
plt.legend(loc="best")
plt.xlabel("F(x)")
#plt.ylim((1e-5,1e20))
plt.ylabel("x")
plt.title("GRAFICA DE LOS ERRORES FUNCIONES POR MEDIO DE LA INTERPOOLACION POLINOMICA")
plt.grid()
plt.show()
