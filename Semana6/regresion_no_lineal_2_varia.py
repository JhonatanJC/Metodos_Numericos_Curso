# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 04:33:28 2020

@author: pc
"""

import numpy as np
x_prueba=[0.25,0.75,1.25,1.75,2.25]
y_prueba=[0.28,0.57,0.68,0.74,0.79]
x_prueba2=[-1,0,1,2,3]
y_prueba2=[1.1,2,3.4,5,9]

def f(x,a_0,a_1):
    return a_0*np.exp(a_1*x)#a_0*(1-np.exp(-a_1*x))
def f_der_a_0(x,a_0,a_1):
    h=1e-5
    return (f(x,a_0 + h,a_1) - f(x,a_0 - h,a_1))/(2*h)
def f_der_a_1(x,a_0,a_1):
    h=1e-5
    return (f(x,a_0 ,a_1+h) - f(x,a_0 ,a_1-h))/(2*h)
#En caso de teneer otra funcion con una coeficiente mas habilitar lo comentado
#def f_der_a_1(x,a_0,a_1):
#   h=1e-5
#   return (f(x,a_0 ,a_1+h) - f(x,a_0 ,a_1-h))/(2*h)
def regrecion_NO_lineal(datos_x,datos_y,f,n,tolerancia):#n:numero de coeficientes a estimar
    """ ver: https://astro.temple.edu/~dhill001/course/NUMANAL_FALL2016/Section%20Lectures/Gauss-Newton%20Nonlinear%20RegressionREVISED.pdf"""
    num_iter_max=100
    soluciones=[np.ones(n)]
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
                #En caso de tener una funcion con un coeficiente mas
                #elif j==1:
                #    A[i][j]=f_der_a_1(x[i],soluciones[k][0],soluciones[k][1])
                
        for j in range(x.shape[0]):
            b[j]=y[j]-f(x[j],soluciones[k][0],soluciones[k][1])
        new_A=A.T@A
        new_b=A.T@b
        solution1=np.linalg.solve(new_A,new_b)
        solucion=soluciones[k] + solution1
        soluciones.append(solucion)
        errores.append(np.linalg.norm(solucion-soluciones[k])/np.linalg.norm(solucion))
        k=k+1
    return soluciones
print(regrecion_NO_lineal(x_prueba2,y_prueba2,f,2,1e-10))
