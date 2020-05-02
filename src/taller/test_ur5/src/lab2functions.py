import numpy as np
from copy import copy

cos=np.cos; sin=np.sin; pi=np.pi


def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.

    """

    T = np.array([[cos(theta), -cos(alpha)*sin(theta),  sin(alpha)*sin(theta), a*cos(theta)],
                  [sin(theta),  cos(alpha)*cos(theta), -sin(alpha)*cos(theta), a*sin(theta)],
                  [         0,             sin(alpha),             cos(alpha),            d],
                  [         0,                      0,                      0,            1]])
    return T
    
    

def fkine_ur5(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]

    """

    # Matrices DH
    T1 = dh(  0.0892,      q[0],      0,  pi/2)
    T2 = dh(  0.1093,   q[1]+pi,  0.425,     0)
    T3 = dh( -0.1093,      q[2],  0.392,     0)
    T4 = dh(  0.1093,   q[3]+pi,      0,  pi/2)
    T5 = dh( 0.09475,   q[4]+pi,      0,  pi/2)
    T6 = dh(  0.0825,      q[5],      0,    pi)
    
    # Efector final con respecto a la base
    T = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)

    return T


def jacobian_ur5(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """
    # Alocacion de memoria
    J = np.zeros((3,6))
    # Transformacion homogenea inicial (usando q)
    T = fkine_ur5(q)

    # Iteracion para la derivada de cada columna
    
    for i in xrange(6):
        # Copiar la configuracion articular inicial
        dq = copy(q);
        # Incrementar la articulacion i-esima usando un delta
        dq[i] += delta
        # Transformacion homogenea luego del incremento (q+dq)
        T2 = fkine_ur5(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[0][i] = (T2[0][3] - T[0][3])/delta
        J[1][i] = (T2[1][3] - T[1][3])/delta
        J[2][i] = (T2[2][3] - T[2][3])/delta

    return J


def ikine_ur5(xdes, q0):
    epsilon  = 1e-18
    max_iter = 100
    delta    = 0.00001

    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian_ur5(q, delta)
        f = fkine_ur5(q)
        xa = f[np.arange(3),3]
        e = xdes-xa
        q = q + np.dot(np.linalg.pinv(J), e)
        # Condicion de termino
        if (np.linalg.norm(np.round(e,5)) < epsilon):
            return q

    return False