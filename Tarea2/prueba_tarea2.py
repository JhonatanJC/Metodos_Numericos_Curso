#! /usr/bin/python3

import numpy as np

import matplotlib.pyplot as plt # from pylab import plot,show

import warnings
warnings.filterwarnings("ignore")

x=np.linspace(-3,3,1000)

def f(x):
    return 5*np.exp(-x) + x - 5
def g(x):
    return x**3+3*x**2+2*x

def busqueda_incremental(x_0,paso,iteraciones,funcion):
    i=0
    intervalos=[]
    print("Intervalo de busqueda: [{},{}] , con paso= {}".format(x_0,x_0+paso*iteraciones, paso))
    while i<=iteraciones:
        x_1=x_0+paso
        if funcion(x_1)*funcion(x_0)<0:
            if x_0>x_1:
                r=(x_1,x_0)
            else:
                r=(x_0,x_1)
            intervalos.append(r)
        x_0=x_1
        i=i+1
    intervalos=np.array(intervalos)
    return intervalos
    
#print(busqueda_incremental(-5,0.0001,100000,f))

def metodo_biseccion(intervalos,iteraciones,tolerancia,funcion):
    raices=[]
    for interv_de_raiz in intervalos:
        i=0
        error_final=10
        x_l=interv_de_raiz[0]
        x_u=interv_de_raiz[1]
        errores=[0]
        while i<=iteraciones and tolerancia>=error_final:
            x_r=(x_l+x_u)/2.0
            errores.append(x_r) 
            if funcion(x_l)*funcion(x_r)<0:
                x_u=x_r
            elif funcion(x_l)*funcion(x_r)>0:
                x_l=x_r
            else:
                break
            i=i+1
        print("Para {} iteraciones".format(i))
        error_final=abs(errores[-1] - errores[-2])
        #print(errores)
        print("error final: ", error_final)
        raices.append(x_r)
    return raices 

def metodo_falsa_posicion(intervalos,iteraciones,tolerancia,funcion):
    raices=[]
    for interv_de_raiz in intervalos:
        i=0
        error_final=10
        x_l=interv_de_raiz[0]
        x_u=interv_de_raiz[1]
        errores=[0]
        while i<=iteraciones and tolerancia>=error_final:
            x_r=x_u - (funcion(x_u)*(x_l-x_u))/(funcion(x_l)-funcion(x_u))
            errores.append(x_r) 
            if funcion(x_l)*funcion(x_r)<0:
                x_u=x_r
            elif funcion(x_l)*funcion(x_r)>0:
                x_l=x_r
            else:
                break
            i=i+1
            #print(errores)
        #print(errores)
        error_final=abs(errores[-1] -errores[-2])
        print("Para {} iteraciones".format(i))
        print("error final: ", error_final)
        raices.append(x_r)
    return raices 

print("INTERVALOS")
intervalos= busqueda_incremental(-10,0.3,1000,f)      
print(intervalos)
print("\n")
print("El numero de raices en este intervalor es {}".format(len(intervalos)))
print("\n")
print("Busqueda Incremental")
print("Soluciones: ",np.mean(busqueda_incremental(-5,0.000001,10**(-6),f),axis=1))
print("\n")
print("Biseccion")
print("Soluciones: ",metodo_biseccion(intervalos,10,10**(-6),f))
print("\n")
print("falsa posicion")
print("Soluciones: ",metodo_falsa_posicion(intervalos,100,10**(-6),f))

"""
plt.plot(x,g(x))
plt.grid()
plt.show()
"""
