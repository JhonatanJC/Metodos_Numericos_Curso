import numpy as np
import matplotlib.pyplot as plt


#x=[1,2,3,4,5,6,7]
x=[1,2,3,4,5,6,7,8,9]
#y=[0.5,2.5,2.0,4.0,3.5,6.0,5.5]
y=[11.755556,10.538889,9.322222,8.105556,6.888889,5.672222,4.455556,3.238889,2.022222]
###################################################
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
    #a_0=(sumx2*sumy-sumx*sumxy)/(n*sumx2 - sumx**2) #Otra forma de hallar del a_0

    S_r=sum([(y[i]-a_0-a_1*x[i])**2 for i in range(len(y))])
    S_t=sum([(y[i]-np.mean(y))**2 for i in range(len(y))])

    S_y_x=((1/(n-2))*S_r)**(0.5) # Error de f(x) debido al ajuste
    S_y=((1/(n-1))*S_t)**(0.5)  #Desviación estandar estadistica con respecto a los datos

    Error_a_0=S_y_x*(sumx2/(n*sumx2 - sumx**2))**(0.5) #Error de a_0
    Error_a_1=S_y_x*(n/(n*sumx2 - sumx**2))**(0.5)     #Error de a_1
    
    #Otra manera de hallar el R2
    #sigmax=np.sqrt(sumx2/n - (sumx/n)**2 )
    #sigmay = np.sqrt(sumy2/n - (sumy/n)**2 )
    #sigmaxy = sumxy/n - (sumx/n)*(sumy/n)
    #R2 = (sigmaxy/(sigmax*sigmay))**2

    R2 = (S_t - S_r)/S_t
    return a_0,a_1,S_y_x,Error_a_0,Error_a_1,R2

regresion1=regresion_lineal(x,y)
print("El valor de a_1 es ", regresion1[1] ," y el valor de a_0 es ", regresion1[0] )
print("La incertidumbre en Y es:",regresion1[2])
print("La incertidumbre en a_0 es:",regresion1[3])
print("La incertidumbre en a_1 es:",regresion1[4])
print("R2",regresion1[5])

#Grafica y juse para todos los datos
with plt.style.context("seaborn-whitegrid"):  
    plt.figure(figsize=(6,4))
    plt.plot(x,y,"ro", label="Datos experimentales")
    plt.plot(x,regresion1[1]*np.array(x)+regresion1[0],label= 'Ajuste de recta')   
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()

########################################
def matriz_positiva(matriz):
    if np.all(np.linalg.eigvals(matriz) > 0) and np.all(matriz.T==matriz):
        return True 
    else:
        return False
def grad_desc_conjugado_mejorado(matriz,vector,num_iter_max,tolerancia):
    n = matriz.shape[0]
    k= 0
    soluciones=[np.zeros(n)]
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

#######################################
x2=[0,1,2,3,4,5]
y2=[2.1,7.7,13.6,27.2,40.9,61.1]

def regresion_polinomial(datos_x,datos_y,m): # m:grados del polinomio al que se quiere ajustar
    #ver Numerical Methods in Engineering with Python 3 , Jaan  Kiusalaas, pag 131
    x=np.array(datos_x)
    y=np.array(datos_y)
    n=x.shape[0] #Numero de datos experimentales
    matriz1=np.zeros((m+1,m+1))
    vector=np.zeros(m+1)
    for j in range(m+1):
        matriz1[j][j]=sum(np.power(x,2*j))
    matriz2=np.zeros((m+1,m+1))
    for j in range(m+1):
        for i in range(j+1,m+1):
            matriz2[i][j]= sum(np.power(x,i+j))
    matriz=matriz1 + matriz2.T + matriz2 
    for j in range(m+1):
        vector[j]=sum(np.power(x,j)*y)
    solucion=grad_desc_conjugado_mejorado(matriz,vector,1000,1e-6)[0]
    
    def x_exp(x,m):
        return np.array([x**i for i in range(m+1)])

    S_r=sum([(y[i] - sum(solucion*x_exp(x[i],m)))**2 for i in range(n)])
    S_t=sum([(y[i]-np.mean(y))**2 for i in range(n)])

    S_y_x=((1/(n-(m+1)))*S_r)**(0.5) # Error de f(x) debido al ajuste
    
    R2 = (S_t - S_r)/S_t

    return solucion, R2
x_2=[-0.04,0.93,1.95,2.9,3.83,5,5.98,7.05,8.21,9.08,10.09]
y_2=[-8.66,-6.44,-4.36,-3.27,-0.88,0.87,3.31,4.63,6.19,7.4,8.85]
print(regresion_polinomial(x_2,y_2,1))




