import numpy as np

def Gamma_a(alpha, v):
    """Gamma_alpha function as described in the paper"""
    if alpha == 1:
        return np.exp(-v)
    if (1+(1-alpha)*v)**(1/(alpha-1)) >= 0 :
        return min(1, (1+(1-alpha)*v)**(1/(alpha-1)))
    return 1

def Gamma_inv(alpha, q):
    """Gamma_alpha^{-1} function as described in the paper"""
    if alpha == 1:
        return -np.log(q)
    if q == 0:
        return np.inf
    return (q**(alpha-1)-1)/(1-alpha)

def G_a(alpha, v, p1, q1, p2, q2):
    """G_alpha function as described in the paper"""
    if q1 == 0 and q2 == 0:
        return 0
    return q1*Gamma_a(alpha, Gamma_inv(alpha, q2/q1)*(v-p1)/(p2-p1))

