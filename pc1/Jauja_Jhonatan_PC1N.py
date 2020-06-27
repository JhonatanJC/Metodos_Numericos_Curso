#! /usr/bin/python3

import numpy as np

import matplotlib.pyplot as plt #from pylab import plot,show

import warnings
warnings.filterwarnings("ignore")

###########################################################
#En esta parte se implementaron las funciones para la resolucion de problemas de sistemas de ecuaciones lineales.
###########################################################

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

def Gaus_elimination_to_backward(matriz,vector):
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    if np.trace(matriz)==0: 
        matriz=pivoteo_parcial(matriz,vector)[0]    # Se implementa el pivoteo en casa que sea necesario
        vector=pivoteo_parcial(matriz,vector)[1]
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
    if np.trace(matriz)==0: 
        matriz=pivoteo_parcial(matriz,vector)[0]    # Se implementa el pivoteo en casa que sea necesario
        vector=pivoteo_parcial(matriz,vector)[1]
    for j in range(n-1,-1,-1):
        vector[j]=vector[j]/matriz[j][j]#EN ESTA PARTE HACEMOS 1 A LA DIAGONAL PRINCIPAL
        matriz[j]=matriz[j]/matriz[j][j]
        for i in range(j-1,-1,-1):
            vector[i]=vector[i] - matriz[i][j]*vector[j] # aplicando la eliminacion de gaus, restando primera fila por las siguientes hasta hacer ceroo la primera columna menos el primer elemento, luego restando la segunda fila por las filas restantes de abajo hasta hacer ceros los numeros a partir de esa columna hacia bajo.
            matriz[i]=matriz[i] - matriz[i][j]*matriz[j]
    return matriz,vector

def descompo_LU(matriz,vector): #APLICANDO EL ALGORITMO DE CROUT
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
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

    b=forward(L,vector)     #Se aplican los pasos finales para dar con la solucion
    solucion=backward(U,b)   
    return L,U,solucion


def descomp_cholesky(matriz,vector):
    n=matriz.shape[0]
    matriz=matriz.astype(float)
    vector=vector.astype(float)
    if np.all(np.linalg.eigvals(matriz) > 0) and np.all(matriz.T==matriz):#Se hacen las comprobaciones antes de hacer la descomposicion
        L_cho=np.zeros([n,n])
        for j in range(n):
            suma=0
            for p in range(0,j): # Se deben ejecutar n-1 iteraciones según la formula
                suma =suma+(L_cho[j][p])**2
            L_cho[j][j]=np.sqrt(matriz[j][j] - suma)
            for i in range(n):
                suma=0
                for p in range(0,i):
                    suma=suma + L_cho[j][p]*L_cho[i][p]
                L_cho[i][j]=(matriz[i][j] - suma)/L_cho[j][j]
            
            for i in range(0,j):    #Este bucle hace ceros la parte superior porque es una matriz triangular inferior.
                L_cho[i][j]=0
        U_cho=L_cho.T
        b=forward(L_cho,vector)     #Se aplican los pasos finales para dar con la solucion
        solucion=backward(U_cho,b)   
        return L_cho,U_cho,solucion
    else:
        return "No se puede hacer la descomposicion Cholesky."

###########################################################
#En esta parte se implementaron las funciones para la solucion de ecuaciones no lineales
###########################################################

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
###########################################################
#En esta parte se implementaron las resulucion a los problemas
###########################################################

#PREGUNTA 1
print("#########")
print("Pregunta 1")
print("#########")
Matriz_prob1=np.array([[4,-1,-1,-1],[-1,3,0,-1],[-1,0,3,-1],[-1,-1,-1,4]])
vector_prob1=np.array([5,0,5,0])
print("Mostramos la matriz dada, para hacer el calculo")
print(Matriz_prob1)
print("\nY el vector:")
print(vector_prob1)

print("\nSOLUCION POR MEDIO DE LOS DIFERENTES ALGORITMOS Y METODOS:\n")
print("ELIMINACION GAUSIANA PARA FORWARD Y BACKWARD")
print("\nMAtriz luego de la eliminacion gausina para forward:\n",Gaus_elimination_to_forward(Matriz_prob1,vector_prob1)[0])
print("\nVector luego de la eliminacion gausiana para forward:\n",Gaus_elimination_to_forward(Matriz_prob1,vector_prob1)[1])
print("\nSolucion luego de aplicar algoritmo de forward:", forward(Gaus_elimination_to_forward(Matriz_prob1,vector_prob1)[0],Gaus_elimination_to_forward(Matriz_prob1,vector_prob1)[1]))

print("\nMAtriz luego de la eliminacion gausina para backward:\n",Gaus_elimination_to_backward(Matriz_prob1,vector_prob1)[0])
print("\nVector luego de la eliminacion gausiana para backward:\n",Gaus_elimination_to_backward(Matriz_prob1,vector_prob1)[1])
print("\nSolucion luego de aplicar algoritmo de backward:", backward(Gaus_elimination_to_backward(Matriz_prob1,vector_prob1)[0],Gaus_elimination_to_backward(Matriz_prob1,vector_prob1)[1]))

print("\nDESCOMPOSICION LU(ALGORITMO DE CROUT)")

print("\nMatriz L:\n",descompo_LU(Matriz_prob1,vector_prob1)[0])
print("Matriz U:\n",descompo_LU(Matriz_prob1,vector_prob1)[1])
print("\nSolucion solucion final por descomposición LU :\n", descompo_LU(Matriz_prob1,vector_prob1)[2])

print("\nDESCOMPOSICION CHOLESKY")
print("\nMatriz G:\n",descomp_cholesky(Matriz_prob1,vector_prob1)[0])
print("\nMatriz G(Transpuesta):\n",descomp_cholesky(Matriz_prob1,vector_prob1)[1])
print("\nSolucion solucion final por descomposicion Cholesky :\n", descomp_cholesky(Matriz_prob1,vector_prob1)[2])

#PREGUNTA 2
#Segun el problema construiremos la matriz para resolver el sistema de ecuaciones
print("\n\n#########")
print("Pregunta 2")
print("#########\n")
k1=k2=k3=k4=0.5
m1=2
m2=0.5
m3=0.3
Matriz_prob2=np.array([[(k1+k2)/m1,-k2/m1,0],[-k2/m2,(k2+k3)/m2,-k3/m2],[0,-k3/m3,(k3+k4)/m3]])
vector_prob2=np.array([-1,1.2,1.3])
print("Mostramos la matriz dada, para hacer el calculo")
print(Matriz_prob2)
print("\nY el vector:")
print(vector_prob2)
print("\nNo obstante, notamos que no es simetrica , por lo que no se podrá aplicar descomposicion CHolesky")
print("La manera mas sencilla de resolver esto es multiplicamos por 4 a la primera fila y por 3/5 a la tercera fila de la matriz para hacerla simetrica:")

Matriz_prob2[0]=4*Matriz_prob2[0]
vector_prob2[0]=4*vector_prob2[0]

Matriz_prob2[2]=(3.0/5.0)*Matriz_prob2[2]
vector_prob2[2]=(3.0/5.0)*vector_prob2[2]

print("Nueva matriz:")
print(Matriz_prob2)
print("\nNuevo vector")
print(vector_prob2)

print("AHora si podemos aplicarle los metodos ")
print("\nSOLUCION POR MEDIO DE LOS DIFERENTES ALGORITMOS Y METODOS:\n")
print("ELIMINACION GAUSIANA PARA FORWARD Y BACKWARD")
print("\nMAtriz luego de la eliminacion gausina para forward:\n",Gaus_elimination_to_forward(Matriz_prob2,vector_prob2)[0])
print("\nVector luego de la eliminacion gausiana para forward:\n",Gaus_elimination_to_forward(Matriz_prob2,vector_prob2)[1])
print("\nSolucion luego de aplicar algoritmo de forward:", forward(Gaus_elimination_to_forward(Matriz_prob2,vector_prob2)[0],Gaus_elimination_to_forward(Matriz_prob2,vector_prob2)[1]))

print("\nMAtriz luego de la eliminacion gausina para backward:\n",Gaus_elimination_to_backward(Matriz_prob2,vector_prob2)[0])
print("\nVector luego de la eliminacion gausiana para backward:\n",Gaus_elimination_to_backward(Matriz_prob2,vector_prob2)[1])
print("\nSolucion luego de aplicar algoritmo de backward:", backward(Gaus_elimination_to_backward(Matriz_prob2,vector_prob2)[0],Gaus_elimination_to_backward(Matriz_prob2,vector_prob2)[1]))

print("\nDESCOMPOSICION LU(ALGORITMO DE CROUT)")

print("\nMatriz L:\n",descompo_LU(Matriz_prob2,vector_prob2)[0])
print("Matriz U:\n",descompo_LU(Matriz_prob2,vector_prob2)[1])
print("\nSolucion solucion final por descomposición LU :\n", descompo_LU(Matriz_prob2,vector_prob2)[2])

print("\nDESCOMPOSICION CHOLESKY")
print("\nMatriz G:\n",descomp_cholesky(Matriz_prob2,vector_prob2)[0])
print("\nMatriz G(Transpuesta):\n",descomp_cholesky(Matriz_prob2,vector_prob2)[1])
print("\nSolucion solucion final por descomposicion Cholesky :\n", descomp_cholesky(Matriz_prob2,vector_prob2)[2])




#PREGUNTA 3
#Segun el problema construiremos la matriz para resolver el sistema de ecuaciones
print("\n\n#########")
print("Pregunta 3")
print("#########")

print("\n#########")
print("Pregunta 3, B")
print("#########\n")

def funcionFx(x):
    Q=1e-4
    q=2e-5
    a=2
    e0=8.8541878176e-12
    return (x*Q*q*np.sin(4*np.pi**2))/(4*np.pi*e0*pow(a**2+x**2,1.5))

def funcionFy(x):
    Q=1e-4
    q=2e-5
    a=2
    e0=8.8541878176e-12
    return (-2*Q*q*a*0.738)/(4*np.pi*e0*pow(a**2+x**2,1.5))

def F(x):
    return pow(funcionFx(x)**2+funcionFy(x)**2,0.5)

x1=np.linspace(-10,10,1000)
x1_F=np.linspace(0,10,1000)

plt.figure()
plt.plot(x1,funcionFx(x1),"r",label="$F_{x}$")
plt.plot(x1,funcionFy(x1),"g",label="$F_{y}$")
plt.plot(x1_F,F(x1_F),"b",label="$F$")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("Fuerza eléctrica")
plt.title("Grafica de las componentes Fx y Fy ademas de la magniud de la fuerza electrica F para un rango de valores entre -10 y 10 (Pregunta 3 b) ")
plt.grid()
plt.show()

print("\n\n#########")
print("Pregunta 3, C")
print("#########\n")
print("Definiremos una nueva función g(x) de la cual hallaremos el valor de la distancia de x, cuando Fx=1.56N: ")

def new_funcion_Fx(x):
    return 1.56 - funcionFx(x)

plt.figure()
plt.plot(x1,funcionFx(new_funcion_Fx(x1)),"r",label="$g(x) = 1.56 - F_{x}$")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("g(x)")
plt.title("Función g(x) de la cual hallaremos sus raices(Pregunta 3 c)")
plt.grid()
#plt.show()


print("\n Notamos de la imagen donde se muestra la funcion para el calculo de x, que hay 2 raices ")
#PRIMERA SOLUCION
metodo_biseccion1=metodo_biseccion(0,1.5,10**(-6),1000,new_funcion_Fx)

metodo_falsa_posicion1=metodo_falsa_posicion(1,1.5,10**(-6),1000,new_funcion_Fx)

metodo_newton_rhapson_modificado1=metodo_newton_rhapson_modificado(1,10**(-6),1000,new_funcion_Fx)

metodo_secante_modificado1=metodo_secante_modificado(1,10**(-6),1000,new_funcion_Fx)

print("\nSe realizó el mismo problema pero con diferente algoritmos para hallar la raiz")
print("Para cada caso se muestra la solución de la funcion g(x), el error con el que se hallo, el numero de iteraciones totales respectivamente:\n")
print("Por ultimo se muestra el valor de h para cada metodo:\n")

print("\nMetodo de la bisecion")
print(metodo_biseccion1[1])
print("Para este caso el valor de x es: ",metodo_biseccion1[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_biseccion1[0])

print("\nMetodo de la falsa posición")
print(metodo_falsa_posicion1[1])
print("Para este caso el valor de x es: ",metodo_falsa_posicion1[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_falsa_posicion1[0])

print("\nMetodo de newton rhapson modificado")
print(metodo_newton_rhapson_modificado1[1])
print("Para este caso el valor de x es: ",metodo_newton_rhapson_modificado1[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_newton_rhapson_modificado1[0])

print("\nMetodo de la secante modificado")
print(metodo_secante_modificado1[1])
print("Para este caso el valor de x es: ",metodo_secante_modificado1[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_secante_modificado1[0])

plt.figure()
plt.plot(np.arange(metodo_biseccion1[1][-1]),np.log(np.array(metodo_biseccion1[0])),label="Biseccion",marker="o",color="blue")
plt.plot(np.arange(metodo_falsa_posicion1[1][-1]),np.log(np.array(metodo_falsa_posicion1[0])),label="Falsa posicion",marker="o",color="green")
plt.plot(np.arange(metodo_newton_rhapson_modificado1[1][-1]),np.log(np.array(metodo_newton_rhapson_modificado1[0])),label="Newton Rhapson modif",marker="o",color="red")
plt.plot(np.arange(metodo_secante_modificado1[1][-1]),np.log(np.array(metodo_secante_modificado1[0])),label="Secante modificada",marker="o",color="yellow")

plt.legend(loc="best")
plt.xlabel("Iteracion")
plt.ylabel("Errores")
plt.title("EVOLUCION DE ERRORES DE CADA METODO NUMERICO EN ESCALA LOGARITMICA PARA LA MENOR RAIZ (Pregunta 3 c)")
plt.grid()
#plt.show()

#PARA LA OTRA SOLUCION

metodo_biseccion2=metodo_biseccion(1.5,3,10**(-6),1000,new_funcion_Fx)

metodo_falsa_posicion2=metodo_falsa_posicion(1.5,3,10**(-6),1000,new_funcion_Fx)

metodo_newton_rhapson_modificado2=metodo_newton_rhapson_modificado(3,10**(-6),1000,new_funcion_Fx)

metodo_secante_modificado2=metodo_secante_modificado(3,10**(-6),1000,new_funcion_Fx)

print("\nSe realizó el mismo problema pero con diferente algoritmos para hallar la raiz")
print("Para cada caso se muestra la solución de la funcion f(x), el error con el que se hallo, el numero de iteraciones totales respectivamente:\n")
print("Por ultimo se muestra el valor de h para cada metodo:\n")

print("\nMetodo de la bisecion")
print(metodo_biseccion2[1])
print("Para este caso el valor de h es: ",metodo_biseccion2[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_biseccion2[0])


print("\nMetodo de la falsa posición")
print(metodo_falsa_posicion2[1])
print("Para este caso el valor de h es: ",metodo_falsa_posicion2[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_falsa_posicion2[0])


print("\nMetodo de newton rhapson modificado")
print(metodo_newton_rhapson_modificado2[1])
print("Para este caso el valor de h es: ",metodo_newton_rhapson_modificado2[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_newton_rhapson_modificado2[0])


print("\nMetodo de la secante modificado")
print(metodo_secante_modificado2[1])
print("Para este caso el valor de x es: ",metodo_secante_modificado2[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_secante_modificado2[0])


plt.figure()
plt.plot(np.arange(metodo_biseccion2[1][-1]),np.log(np.array(metodo_biseccion2[0])),label="Biseccion",marker="o",color="blue")
plt.plot(np.arange(metodo_falsa_posicion2[1][-1]),np.log(np.array(metodo_falsa_posicion2[0])),label="Falsa posicion",marker="o",color="green")
plt.plot(np.arange(metodo_newton_rhapson_modificado2[1][-1]),np.log(np.array(metodo_newton_rhapson_modificado2[0])),label="Newton Rhapson modif",marker="o",color="red")
plt.plot(np.arange(metodo_secante_modificado2[1][-1]),np.log(np.array(metodo_secante_modificado2[0])),label="Secante modificada",marker="o",color="yellow")

plt.legend(loc="best")
plt.xlabel("Iteracion")
plt.ylabel("Errores")
plt.title("EVOLUCION DE ERRORES DE CADA METODO NUMERICO EN ESCALA LOGARITMICA PARA LA MAYOR RAIZ(Pregunta 3 c)")
plt.grid()
plt.show()





print("\n\n#########")
print("Pregunta 3, D")
print("#########\n")

def new_funcion_Fy(x):
    return -2.17 - funcionFy(x)

plt.figure()
plt.plot(x1,funcionFx(new_funcion_Fy(x1)),"r",label="$h(x)=-2.17 - F_{y}$")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("h(x)")
plt.title("Función h(x) de la cual hallaremos sus raices(Pregunta 3 c)")
plt.grid()
#plt.show()


print("\n Notamos de la imagen donde se muestra la funcion para el calculo de x, que hay 2 raices , pero solo consideramos la raiz positiva:")

metodo_biseccion1=metodo_biseccion(0,5,10**(-6),1000,new_funcion_Fy)

metodo_falsa_posicion1=metodo_falsa_posicion(0,5,10**(-6),1000,new_funcion_Fy)

metodo_newton_rhapson_modificado1=metodo_newton_rhapson_modificado(2,10**(-6),1000,new_funcion_Fy)

metodo_secante_modificado1=metodo_secante_modificado(1,10**(-6),1000,new_funcion_Fy)

print("\nSe realizó el mismo problema pero con diferente algoritmos para hallar la raiz")
print("Para cada caso se muestra la solución de la funcion f(x), el error con el que se hallo, el numero de iteraciones totales respectivamente:\n")
print("Por ultimo se muestra el valor de h para cada metodo:\n")

print("\nMetodo de la bisecion")
print(metodo_biseccion1[1])
print("Para este caso el valor de h es: ",metodo_biseccion1[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_biseccion1[0])

print("\nMetodo de la falsa posición")
print(metodo_falsa_posicion1[1])
print("Para este caso el valor de h es: ",metodo_falsa_posicion1[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_falsa_posicion1[0])

print("\nMetodo de newton rhapson modificado")
print(metodo_newton_rhapson_modificado1[1])
print("Para este caso el valor de h es: ",metodo_newton_rhapson_modificado1[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_newton_rhapson_modificado1[0])

print("\nMetodo de la secante modificado")
print(metodo_secante_modificado1[1])
print("Para este caso el valor de x es: ",metodo_secante_modificado1[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_secante_modificado1[0])

plt.figure()
plt.plot(np.arange(metodo_biseccion1[1][-1]),np.log(np.array(metodo_biseccion1[0])),label="Biseccion",marker="o",color="blue")
plt.plot(np.arange(metodo_falsa_posicion1[1][-1]),np.log(np.array(metodo_falsa_posicion1[0])),label="Falsa posicion",marker="o",color="green")
plt.plot(np.arange(metodo_newton_rhapson_modificado1[1][-1]),np.log(np.array(metodo_newton_rhapson_modificado1[0])),label="Newton Rhapson modif",marker="o",color="red")
plt.plot(np.arange(metodo_secante_modificado1[1][-1]),np.log(np.array(metodo_secante_modificado1[0])),label="Secante modificada",marker="o",color="yellow")

plt.legend(loc="best")
plt.xlabel("Iteracion")
plt.ylabel("Errores")
plt.title("EVOLUCION DE ERRORES DE CADA METODO NUMERICO EN ESCALA LOGARITMICA PARA LA RAIZ POSITIVA(Pregunta 3 d)")
plt.grid()
plt.show()

#PREGUNTA 4
#Segun el problema construiremos la matriz para resolver el sistema de ecuaciones
print("\n\n#########")
print("Pregunta 4")
print("#########\n")

def funcion_preg4(h):
    densidad_agua=1000
    densidad_s=200
    r=1
    return 4*densidad_s*r**3 - densidad_agua*4*r**3 +densidad_agua*(3*r-h)*h**2

h=np.linspace(-1.5,3,1000)
plt.figure()
plt.plot(h,funcion_preg4(h),"r",label="$f(h) = \\rho_{s} { r }^{3}- \\rho_{H_{2}O}4 { r }^{3} + \\rho_{H_{2}O}{ h }^{2}(3r-h)$")
plt.legend(loc="best")
plt.xlabel("h")
plt.ylabel("f(h)")
plt.title("Funcion para el calculo de h, con r=1m $\\rho_{s} =200 Kg/m^{3} $ y $\\rho_{H_{2}O}=1000 Kg/m^{3} (Pregunta 4)$")
plt.grid()
#plt.show()

print("\n Notamos de la imagen donde se muestra la funcion para el calculo de h, que hay 3 raices una negativa y 2 positivas")
print("\nComo h es un distancia, entonces descartamos el valor negativo, ademas como 0<h<2r, entonces solo nos centraremos en el valor de h que esta entre 1 y 2")
metodo_biseccion1=metodo_biseccion(1,2,10**(-6),1000,funcion_preg4)

metodo_falsa_posicion1=metodo_falsa_posicion(1,2,10**(-6),1000,funcion_preg4)

metodo_newton_rhapson_modificado1=metodo_newton_rhapson_modificado(1000,10**(-6),1000,funcion_preg4)

print("\nSe realizó el mismo problema pero con diferente algoritmos para hallar la raiz")
print("Para cada caso se muestra la solución de la funcion f(x), el error con el que se hallo, el numero de iteraciones totales respectivamente:\n")
print("Por ultimo se muestra el valor de h para cada metodo:\n")

print("\nMetodo de la bisecion")
print(metodo_biseccion1[1])
print("Para este caso el valor de h es: ",metodo_biseccion1[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_biseccion1[0])

print("\nMetodo de la falsa posición")
print(metodo_falsa_posicion1[1])
print("Para este caso el valor de h es: ",metodo_falsa_posicion1[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_falsa_posicion1[0])

print("\nMetodo de newton rhapson modificado")
print(metodo_newton_rhapson_modificado1[1])
print("Para este caso el valor de h es: ",metodo_newton_rhapson_modificado1[1][0])
print("Y el error en cada iteracion viene dada por el siguiente array:\n")
print(metodo_newton_rhapson_modificado1[0])

plt.figure()
plt.plot(np.arange(metodo_biseccion1[1][-1]),np.log(np.array(metodo_biseccion1[0])),label="Biseccion",marker="o",color="blue")
plt.plot(np.arange(metodo_falsa_posicion1[1][-1]),np.log(np.array(metodo_falsa_posicion1[0])),label="Falsa posicion",marker="o",color="green")
plt.plot(np.arange(metodo_newton_rhapson_modificado1[1][-1]),np.log(np.array(metodo_newton_rhapson_modificado1[0])),label="Newton Rhapson modif",marker="o",color="red")

plt.legend(loc="best")
plt.xlabel("Iteracion")
plt.ylabel("Errores")
plt.title("EVOLUCION DE ERRORES DE CADA METODO NUMERICO EN ESCALA LOGARITMICA PARA LA RAIZ QUE CUMPLA 0<h<2(Pregunta 4)")
plt.grid()
plt.show()


