#! /usr/bin/python3

import numpy as np

import matplotlib.pyplot as plt #from pylab import plot,show

import warnings
warnings.filterwarnings("ignore")

#REGRESION LINEAL
def regresion_lineal(datos_x,datos_y):
    x=np.array(datos_x)
    y=np.array(datos_y)
    n=x.shape[0]
    sumx=sum(x)
    sumy=sum(y)
    sumx2=sum(x*x)
    sumy2=sum(y*y)
    sumxy=sum(x*y)

    a_1=(n*sumxy - sumx*sumy)/(n*sumx2 - sumx**2)
    a_0=np.mean(y)- a_1*np.mean(x)

    S_r=sum([(y[i]-a_0-a_1*x[i])**2 for i in range(len(y))])
    S_t=sum([(y[i]-np.mean(y))**2 for i in range(len(y))])

    S_y_x=((1/(n-2))*S_r)**(0.5) # Error de f(x) debido al ajuste
    S_y=((1/(n-1))*S_t)**(0.5)  #Desviación estandar estadistica con respecto a los datos

    Error_a_0=S_y_x*(sumx2/(n*sumx2 - sumx**2))**(0.5) #Error de a_0
    Error_a_1=S_y_x*(n/(n*sumx2 - sumx**2))**(0.5)     #Error de a_1
    
    R2 = (S_t - S_r)/S_t
    return [a_0,a_1] #a_0,a_1,S_y_x,Error_a_0,Error_a_1,R2

#REGRESION NO LINEAL
#Funcion a ajustar y sus derivadas parciales con respeto a las constante a hallar
def f(x,a_0,a_1):
    return a_0*x*np.exp(a_1*x)#a_0*(1-np.exp(-a_1*x))
def f_der_a_0(x,a_0,a_1):
    h=1e-5
    return (f(x,a_0 + h,a_1) - f(x,a_0 - h,a_1))/(2*h)
def f_der_a_1(x,a_0,a_1):
    h=1e-5
    return (f(x,a_0 ,a_1+h) - f(x,a_0 ,a_1-h))/(2*h)
###
def matriz_positiva(matriz):
    if np.all(np.linalg.eigvals(matriz) > 0) and np.all(matriz.T==matriz):
        return True 
    else:
        return False
def grad_desc_conjugado_mejorado(matriz,vector,num_iter_max,tolerancia):
    n = matriz.shape[0]
    k= 0
    soluciones=[np.ones(n)]
    r=[matriz@soluciones[0] - vector]
    p_s=-r[0]
    errores=[10]
    
    while k<num_iter_max and errores[k] > tolerancia:
        w=matriz@p_s
        alpha=(r[k].T@r[k])/(p_s.T@w) 
        solucion=soluciones[k] + alpha*p_s
        r_s=r[k] + alpha*w
        r.append(r_s)
        beta=(r_s.T@r_s)/(r[k].T@r[k])
        p_s=-r_s+ beta*p_s
        soluciones.append(solucion)
        errores.append(np.linalg.norm(solucion-soluciones[k])/np.linalg.norm(solucion))
        k= k + 1
        soluciones_final=soluciones[-1]
    if matriz_positiva(matriz):
        return soluciones_final,soluciones,errores, k
    else:
        print("La matriz no es positiva y/o simetrica el resultados no puede ser el correcto ")
        return soluciones_final,soluciones,errores, k


###

def regrecion_NO_lineal(datos_x,datos_y,f,n,tolerancia):#n:numero de coeficientes a estimar
    """ ver: https://astro.temple.edu/~dhill001/course/NUMANAL_FALL2016/Section%20Lectures/Gauss-Newton%20Nonlinear%20RegressionREVISED.pdf"""
    num_iter_max=100
    solucion_inicial=np.ones(n)  #SOlucion incial para este caso.
    soluciones=[solucion_inicial]
    x=np.array(datos_x)
    y=np.array(datos_y)
    A=np.zeros((x.shape[0],n))
    b=np.zeros(x.shape[0])
    errores=[1]
    k=0
    while k<num_iter_max and errores[k]>tolerancia:
        for j in range(n):
            for i in range(x.shape[0]):
                if j==0:
                    A[i][j]=f_der_a_0(x[i],soluciones[k][0],soluciones[k][1])
                elif j==1:
                    A[i][j]=f_der_a_1(x[i],soluciones[k][0],soluciones[k][1])
        for j in range(x.shape[0]):
            b[j]=y[j]-f(x[j],soluciones[k][0],soluciones[k][1])
        new_A=A.T@A
        new_b=A.T@b
        solution1=grad_desc_conjugado_mejorado(new_A,new_b,1000,1e-6)[0]
        solucion=soluciones[k] + solution1
        soluciones.append(solucion)
        errores.append(np.linalg.norm(solucion-soluciones[k])/np.linalg.norm(solucion))
        k=k+1
    return soluciones[-1]

###PROBLEMA 1
print("Problema 1:")
datos_x_p1=np.array([0.5,1,2,3,4])
datos_y_p1=np.array([10.4,5.8,3.3,2.4,2])

print("\nDatos de x: ", datos_x_p1)
print("\nDatos de y: ", datos_y_p1)

datos_y_p1_lineal=np.sqrt(datos_y_p1)
datos_x_p1_lineal=1/np.sqrt(datos_x_p1)
print("\nDatos de x, luego de linealizar: ", datos_x_p1_lineal)
print("\nDatos de y: luego de linealizar: ", datos_y_p1_lineal)

regresion_lin_p1=regresion_lineal(datos_x_p1_lineal,datos_y_p1_lineal)

print("\nEntonces los coeficientes a_0 y a_1, son respectivamente:")
print(regresion_lin_p1)
print("\nAhora como a_0= a/b y a_1=1/b")
a=regresion_lin_p1[1]/regresion_lin_p1[0]
b=a/regresion_lin_p1[0]
print("Entonces: a= {} y b= {}".format(a,b))

###PROBLEMA 2
print("\nProblema 2:\n")
print("PRIMERA FORMA, REGRESION LINEAL:")
datos_x_p2=np.array([0.1,0.2,0.4,0.6,0.9,1.3,1.5,1.7,1.8])
datos_y_p2=np.array([0.75,1.25,1.45,1.25,0.85,0.55,0.35,0.28,0.18])

print("\nDatos de x: ", datos_x_p2)
print("\nDatos de y: ", datos_y_p2)

datos_y_p2_lineal=np.log(datos_y_p2/datos_x_p2)
datos_x_p2_lineal=datos_x_p2
print("\nDatos de x, luego de linealizar: ", datos_x_p2_lineal)
print("\nDatos de y: luego de linealizar: ", datos_y_p2_lineal)

regresion_lin_p2=regresion_lineal(datos_x_p2_lineal,datos_y_p2_lineal)

print("\nEntonces los coeficientes a_0 y a_1, son respectivamente:")
print(regresion_lin_p2)
print("\nAhora como a_0= ln(alfa4) y a_1=beta4")
alfa4=np.exp(regresion_lin_p2[0])
beta4=regresion_lin_p2[1]
print("\nEntonces: alfa4= {} y beta4= {}".format(alfa4,beta4))

print("\nSEGUNDA FORMA, REGRESION NO LINEAL:\n")
solucion_inicial=np.ones(2)
print("Se utilizó como solucion inicial: ", solucion_inicial )
regresion_no_lin_p2=regrecion_NO_lineal(datos_x_p2,datos_y_p2,f,2,1e-6)
print("Entonces, los coeficientes calculados por ajuste NO lineal son : alfa4= {} y beta4= {}".format(regresion_no_lin_p2[0],regresion_no_lin_p2[1]))


