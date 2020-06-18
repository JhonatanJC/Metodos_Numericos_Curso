#! /usr/bin/python3

import numpy as np

import matplotlib.pyplot as plt # from pylab import plot,show

import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 16}) #Solo para cambiar el tamañano de las letras y numeros en las graficas

x=np.linspace(-1,6.5,1000)

def f(x):   #Función cuya raiz nos dice el lamnda maximo y  apartir de la cual se calcula la const de Wien
    return 5*np.exp(-x) + x - 5

plt.figure()
plt.plot(x,f(x),"r",label="$f(x) =5{ e }^{ -x }+x-5$")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Función para la determinación de la constante de desplazamiento de Wien")
plt.grid()
plt.show()

def conste_Wine(x): #Esta funcion solo halla b(constante de Wien) a partir de x
    h=6.62607015e-34
    c=299792458.0
    k=1.380649e-23
    return (h*c)/(k*x)

def busqueda_incremental(x_0,paso,itera_max,funcion):
    i=0
    while i<=itera_max:     #la función solo recorrere todos los puntos paso a paso 
        x_1=x_0+paso
        if funcion(x_1)*funcion(x_0)<=0:
            return (x_1+x_0)/2.0
        else:
            x_0=x_1
        i=i+1
    print("NO HAY SOLUCION")

#Para poder hallar los errores consecutivos se guardan estos valores en listas vacias que se van llenando en cada iteracion 
#ESte se utiliza para todas las funciones para asi luego poder graficar la evolucion de estos valores.
def metodo_biseccion(x_l,x_u,tolerancia,iteraciones,funcion):
    i=0
    error_final=10  #errore inicial
    todos_x_r=[0]   #Lista que guarda los valores de la raiz en cada iteracion y tiene 0 como valores incial para poder hallar el primer error 
    errores=[]      #Lista que guarda los valores de las diferencias de raices consecutivas osea los errores en cada iteración esta lista al final se usa para hacer la evolucion de errores
    while i<=iteraciones and tolerancia<error_final:
        x_r=(x_l+x_u)/2.0   #regla de iteracion
        todos_x_r.append(x_r)
        errores.append(abs(todos_x_r[i+1]-todos_x_r[i])/todos_x_r[i+1]) #Se agregan las diferencias de raices consecutivas osea los errores
        if funcion(x_l)*funcion(x_r)<0:
            x_u=x_r
        elif funcion(x_l)*funcion(x_r)>0:
            x_l=x_r
        else:
            break
        i=i+1
        error_final=errores[-1]  #se actualiza el valor de error final, el cual se comprará con la tolerancia 
    return (errores,(todos_x_r[-1],error_final,i)) # retorna una tupla que tiene los errores en cada iteracion, tamben la raiz, el error final y el # de iteaciones. 

#Todas los metodos implementados funcionan parecido solo cambia su regla de proxima iteacion.
def metodo_falsa_posicion(x_l,x_u,tolerancia,iteraciones,funcion):
    i=0
    error_final=10
    todos_x_r=[0]
    errores=[]
    while i<=iteraciones and tolerancia<error_final:
        x_r=x_u - (funcion(x_u)*(x_l-x_u))/(funcion(x_l)-funcion(x_u))
        todos_x_r.append(x_r)
        errores.append(abs(todos_x_r[i+1]-todos_x_r[i])/todos_x_r[i+1])
        if funcion(x_l)*funcion(x_r)<0:
            x_u=x_r
        elif funcion(x_l)*funcion(x_r)>0:
            x_l=x_r
        else:
            break
        i=i+1
        error_final=errores[-1]  
    return (errores,(todos_x_r[-1],error_final,i))

def metodo_punto_fijo(x_0,tolerancia,iteraciones,funcion):
    i=0
    error_final=10
    todos_x_r=[0]
    errores=[]
    while i<=iteraciones and tolerancia<error_final:
        x_1=x_0 - funcion(x_0) 
        todos_x_r.append(x_1)
        errores.append(abs(todos_x_r[i+1]-todos_x_r[i])/todos_x_r[i+1])
        x_0=x_1
        i=i+1
        error_final=errores[-1]
    return (errores,(todos_x_r[-1],error_final,i))


#Derivacion numerica, formulas utilizadas para el metodo de newthion rhapson y tambien para el metodo de la secante
def primera_derivada(f,x):
    h=0.0002#1e-10#0.0002
    """CentralFiniteDifference Approximations"""
    return (f(x+h)-f(x-h))/(2*h)
def segunda_derivada(f,x):
    h=0.0002#1e-10#0.0002
    """CentralFiniteDifference Approximations"""
    return (f(x+h)-2*f(x)+f(x-h))/(h**2)


def metodo_newton_rhapson_modificado(x_0,tolerancia,iteraciones,funcion):
    i=0
    error_final=10
    todos_x_r=[0]
    errores=[]
    while i<=iteraciones and tolerancia<error_final:
        #x_1=x_0 - funcion(x_0)/primera_derivada(funcion,x_0) # NERTHON RHAPSON NORMAL
        x_1 = x_0 - (funcion(x_0)*primera_derivada(funcion,x_0))/(primera_derivada(funcion,x_0)**2 - funcion(x_0)*segunda_derivada(funcion,x_0)) #MODIFICADO
        todos_x_r.append(x_1)
        errores.append(abs(todos_x_r[i+1]-todos_x_r[i])/todos_x_r[i+1])
        x_0=x_1
        i=i+1
        error_final=errores[-1]
    return (errores,(todos_x_r[-1],error_final,i))

def metodo_secante_modificado(x_0,tolerancia,iteraciones,funcion):
    x_01=2
    i=0
    error_final=10
    todos_x_r=[0]
    errores=[]
    while i<=iteraciones and tolerancia<error_final:
        u=funcion(x_0)/primera_derivada(funcion,x_0)
        d=funcion(x_01)/primera_derivada(funcion,x_01)
        x_1=x_0 - (u*(x_01-x_0))/(d-u)    
        todos_x_r.append(x_1)
        errores.append(abs(todos_x_r[i+1]-todos_x_r[i])/todos_x_r[i+1])
        x_01=x_0
        x_0=x_1
        i=i+1
        error_final=errores[-1]
    return (errores,(todos_x_r[-1],error_final,i))

busqueda_incremental1=busqueda_incremental(4.5,10**(-5),10**(6),f)
metodo_biseccion1=metodo_biseccion(1,110,10**(-6),1000,f)
metodo_falsa_posicion1=metodo_falsa_posicion(1,110,10**(-6),1000,f)

metodo_punto_fijo1=metodo_punto_fijo(100,10**(-6),1000,f)
metodo_newton_rhapson_modificado1=metodo_newton_rhapson_modificado(1000,10**(-6),1000,f)
metodo_secante_modificado1=metodo_secante_modificado(1000,10**(-6),1000,f)

print("\nSe realizó el mismo problema pero con diferente algoritmos para hallar la raiz")
print("Para cada caso se muestra la solución de la funcion f(x), el error con el que se hallo, el numero de iteraciones totales respectivamente:\n")
print("Por ultimo se muestra la constante de Wein para cada metodo:\n")
print("Metodo de busqueda incremental")
print(busqueda_incremental1)
print("Para este caso el valor de la constante de Wein es: ",conste_Wine(busqueda_incremental1))

print("\nMetodo de la bisecion")
print(metodo_biseccion1[1])
print("Para este caso el valor de la constante de de Wein es: ",conste_Wine(metodo_biseccion1[1][0]))

print("\nMetodo de la falsa posición")
print(metodo_falsa_posicion1[1])
print("Para este caso el valor de la constante de Wein es: ",conste_Wine(metodo_falsa_posicion1[1][0]))

print("\nMetodo del punto fijo")
print(metodo_punto_fijo1[1])
print("Para este caso el valor de la constante de Wein es: ",conste_Wine(metodo_punto_fijo1[1][0]))

print("\nMetodo de newton rhapson modificado")
print(metodo_newton_rhapson_modificado1[1])
print("Para este caso el valor de la constante de Wein es: ",conste_Wine(metodo_newton_rhapson_modificado1[1][0]))

print("\nMetodo de la secante modificado")
print(metodo_secante_modificado1[1])
print("Para este caso el valor de la constante de Wein es: ",conste_Wine(metodo_secante_modificado1[1][0]))

plt.figure()
plt.plot(np.arange(metodo_biseccion1[1][-1]),np.log(np.array(metodo_biseccion1[0])),label="Biseccion",marker="o",color="blue")#,s=10)
plt.plot(np.arange(metodo_falsa_posicion1[1][-1]),np.log(np.array(metodo_falsa_posicion1[0])),label="Falsa posicion",marker="o",color="green")#,s=100)
plt.plot(np.arange(metodo_punto_fijo1[1][-1]),np.log(np.array(metodo_punto_fijo1[0])),label="Punto Fijo",marker="o",color="black")#,s=100)
plt.plot(np.arange(metodo_newton_rhapson_modificado1[1][-1]),np.log(np.array(metodo_newton_rhapson_modificado1[0])),label="Newton Rhapson modif",marker="o",color="r")#,s=100)
plt.plot(np.arange(metodo_secante_modificado1[1][-1]),np.log(np.array(metodo_secante_modificado1[0])),label="Secante modificada",marker="o",color="yellow")#,s=100)
plt.legend(loc="best")
plt.xlabel("Iteracion")
plt.ylabel("Errores")
plt.title("EVOLUCION DE ERRORES DE CADA METODO NUMERICO EN ESCALA LOGARITMICA")
plt.grid()
plt.show()
