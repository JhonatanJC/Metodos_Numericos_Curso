
import numpy as np

# El ejemplo se sacó del siguiente pdf : "https://academic.macewan.ca/burok/Stat378/notes/nonlinear.pdf?fbclid=IwAR1m6VozgHEXx6mpJWh7NiQGD0ZY-MnFARTeEP5f5lhRaSFlW4JU8pxLoBw"
x_prueba3=np.arange(0,21,1)
y_prueba3=np.array([3.929,5.308,7.240,9.638,12.866, 17.069,23.192, 31.443,39.818,50.156,62.948,75.995,91.972,105.711,122.775,131.669,150.697,179.323,203.302, 226.542,248.710])
def f(x,a_0,a_1,a_2):
    return a_0*(1./(1.+np.exp(a_1+a_2*x)))#a_0*np.exp(a_1*x)#a_0*(1-np.exp(-a_1*x))
def f_der_a_0(x,a_0,a_1,a_2):
    h=1e-5
    return (f(x,a_0 + h,a_1,a_2) - f(x,a_0 - h,a_1,a_2))/(2*h)

def f_der_a_1(x,a_0,a_1,a_2):
    h=1e-5
    return (f(x,a_0 ,a_1+h,a_2) - f(x,a_0 ,a_1-h,a_2))/(2*h)

def f_der_a_2(x,a_0,a_1,a_2):
    h=1e-5
    return (f(x,a_0 ,a_1,a_2+h) - f(x,a_0 ,a_1,a_2-h))/(2*h)

def regrecion_NO_lineal(datos_x,datos_y,f,n,tolerancia):#n:numero de coeficientes a estimar
    """ ver: https://astro.temple.edu/~dhill001/course/NUMANAL_FALL2016/Section%20Lectures/Gauss-Newton%20Nonlinear%20RegressionREVISED.pdf, para saber como hacer las matrices."""
    num_iter_max=100
    soluciones=[np.array([400,4.61,-0.3])]#Contiene la solucion inicial que se le tiene que dar, ver: "https://academic.macewan.ca/burok/Stat378/notes/nonlinear.pdf?fbclid=IwAR1m6VozgHEXx6mpJWh7NiQGD0ZY-MnFARTeEP5f5lhRaSFlW4JU8pxLoBw" para saber como se calcularon estos puntos iniciales. 
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
                    A[i][j]=f_der_a_0(x[i],soluciones[k][0],soluciones[k][1],soluciones[k][2])
                elif j==1:
                    A[i][j]=f_der_a_1(x[i],soluciones[k][0],soluciones[k][1],soluciones[k][2])
                elif j==2:
                    A[i][j]=f_der_a_2(x[i],soluciones[k][0],soluciones[k][1],soluciones[k][2])
             
        for j in range(x.shape[0]):
            b[j]=y[j]-f(x[j],soluciones[k][0],soluciones[k][1],soluciones[k][2])
        new_A=A.T@A
        new_b=A.T@b
        solution1=np.linalg.solve(new_A,new_b)
        solucion=soluciones[k] + solution1
        soluciones.append(solucion)
        errores.append(np.linalg.norm(solucion-soluciones[k])/np.linalg.norm(solucion))
        k=k+1
    return soluciones[-1]
print(regrecion_NO_lineal(x_prueba3,y_prueba3,f,3,1e-10))
