import numpy as np
import matplotlib.pyplot as plt
##
def matriz_positiva(matriz):
    if np.all(np.linalg.eigvals(matriz) > 0) and np.all(matriz.T==matriz):
        return True 
    else:
        return False
def grad_desc_conjugado(matriz,vector,num_iter_max,tolerancia):
    n = matriz.shape[0]
    k= 0
    soluciones=[np.zeros(n)]
    r=[matriz@soluciones[0] - vector]
    p_s=-r[0]
    errores=[10]
    
    while k<num_iter_max and errores[k] > tolerancia:
        alpha=-(r[k].T@p_s)/(p_s.T@(matriz@p_s)) 
        solucion=soluciones[k] + alpha*p_s
        r_s=matriz@solucion - vector
        r.append(r_s)
        beta=(r_s.T@(matriz@p_s))/(p_s.T@(matriz@p_s))
        p_s=-r_s + beta*p_s
        soluciones.append(solucion)
        errores.append(np.linalg.norm(solucion-soluciones[k])/np.linalg.norm(solucion))
        k= k + 1
     
    if matriz_positiva(matriz):
        return soluciones,errores, k
    else:
        print("La matriz no es positiva y/o simetrica el resultados no puede ser el correcto ")
        return soluciones,errores, k
##Los datos de x tienen que estar ordenados de menor a mayor
def splines_cubico(valores_x,datos_x,datos_y):
    x=np.array(datos_x)
    y=np.array(datos_y)
    n=x.shape[0]
    matriz_sigma=np.zeros([n-2,n-2])
    vector_sigma=np.zeros(n-2)
    h=[]
    for i in range(1,n): #Para hallar h
        h.append(x[i]-x[i-1])

    #Construyendo matriz_simga
    for j in range(1,n-2):#para los elementos diferentes de la diagonal
        matriz_sigma[j][j-1]=h[j]
    matriz_sigma=matriz_sigma.T+matriz_sigma
    for j in range(1,n-1):#para la diagonal de la matriz
        matriz_sigma[j-1][j-1]=2*(h[j-1]+h[j])
    #Ahora contruyendo el vector_sigma
    for i in range(2,n):
        vector_sigma[i-2]=6*((y[i] - y[i-1])/h[i-1] - (y[i-1] - y[i-2])/h[i-2])
     
    soluciones_sigmas=grad_desc_conjugado(matriz_sigma,vector_sigma,1000,1e-6)[0][-1]
    
    sigmas=np.concatenate([np.array([0]),soluciones_sigmas, np.array([0])])
    
    y_interpo_cub=[]
    for p in valores_x:
        m=0
        for k in range(0,n-1):
            if p>x[k] and p<=x[k+1]:
                q=(sigmas[k]/6)*((x[k+1]-p)**3/h[k] - h[k]*(x[k+1]-p)) +\
                    (sigmas[k+1]/6)*((p-x[k])**3/h[k] - h[k]*(p-x[k])) + \
                    y[k]*(x[k+1]-p)/h[k] + y[k+1]*(p-x[k])/h[k]
                y_interpo_cub.append(q)
                m=m+1
        if m==0:
            if p<=x[0]:
                k=0
                q=(sigmas[k]/6)*((x[k+1]-p)**3/h[k] - h[k]*(x[k+1]-p)) +\
                    (sigmas[k+1]/6)*((p-x[k])**3/h[k] - h[k]*(p-x[k])) + \
                    y[k]*(x[k+1]-p)/h[k] + y[k+1]*(p-x[k])/h[k]
                y_interpo_cub.append(q)
            if p>x[n-1]:
                k=n-2
                q=(sigmas[k]/6)*((x[k+1]-p)**3/h[k] - h[k]*(x[k+1]-p)) +\
                    (sigmas[k+1]/6)*((p-x[k])**3/h[k] - h[k]*(p-x[k])) + \
                    y[k]*(x[k+1]-p)/h[k] + y[k+1]*(p-x[k])/h[k]
                y_interpo_cub.append(q)
    return y_interpo_cub

def splines_lineal(valores_x,datos_x,datos_y):
    x=np.array(datos_x)
    y=np.array(datos_y)
    n=x.shape[0]
    y_interpo_lin=[]
    for p in valores_x:
        r=0
        m=0
        for k in range(0,n-1):
            if p>x[k] and p<=x[k+1]:
                m=(y[k+1]-y[k])/(x[k+1]-x[k])
                q=y[k] + m*(p-x[k])
                y_interpo_lin.append(q)
                r=r+1
        if r==0:
            if p<=x[0]:
                k=0
                m=(y[k+1]-y[k])/(x[k+1]-x[k])
                q=y[k] + m*(p-x[k])
                y_interpo_lin.append(q)
            if p>x[n-1]:
                k=n-2
                m=(y[k+1]-y[k])/(x[k+1]-x[k])
                q=y[k] + m*(p-x[k])
                y_interpo_lin.append(q)
    return y_interpo_lin
#Datos de prueba, que se utilizaron para hacer el codigo, ver "http://numat.net/tutor/splines.pdf"
x=[0.1,0.2,0.5,1,2,5,10]
y=[10,5,2,1,0.5,0.2,0.1]


x_s=np.linspace(1,10,1000)

x_tarea=[3,4.5,7,9]
y_tarea=[2.5,1,2.5,0.5]

y_cub=splines_cubico(x_s,x_tarea,y_tarea)
y_lin=splines_lineal(x_s,x_tarea,y_tarea)

plt.scatter(x_tarea,y_tarea)
plt.plot(x_s,y_cub)
plt.plot(x_s,y_lin)
plt.show()
