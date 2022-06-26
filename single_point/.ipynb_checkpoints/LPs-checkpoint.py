from .deterministic import *
from .utils import *

import numpy as np
from scipy.optimize import linprog
from cvxopt import matrix, solvers
solvers.options['show_progress']=False
    
   
def lowerbound_LP_regular(w, q_l, q_h, eta = 1e-5, b = 10, N = 50): 
    """Function that returns a dictionary  {bound, ai_s, mech, sol}
    where bound is the obtained bound from the LP solution,
    ai_s is support of the near-optimal randomized mechanism 
    mech is the weights of the near-optimal randomized mechanism and
    sol is the solution object for the lowerbound LP against Regular distribtions when q in [q_l, q_h]"""
    def F_r(w, q_l, q_h, r, v, left):
        if left:
            if v <= r:
                return 1
            if v <= w:
                return G_a(0, v, r, 1, w, q_l)
            return 0
        else:
            if v <= r:
                return G_a(0, v, 0, 1, w, q_h)
            return 0

    def opt_F_r(w, q_l, q_h, r, left):
        if left:
            return r
        elif r < np.inf:
            return r*G_a(0, r, 0, 1, w, q_h)
        else:
            # lim_{r -> inf} r*G_a(0, r, 0, 1, w, q_h)
            return w/((1/q_h)-1) 

    ai_l = np.linspace(w*q_l, w-eta, N)
    ai_r = np.linspace(w, b, N)
    ai_s = np.append(ai_l, ai_r)
    
    
    N_vars = len(ai_s)
    A_lb = []
    b_lb = []
   
    for i in range(len(ai_l)-1):
        A_lb += [list(np.concatenate((-np.array([ai_s[j]*F_r(w, q_l, q_h, ai_s[i], ai_s[j], left = True) for j in range(len(ai_s))])/opt_F_r(w, q_l, q_h, ai_l[i+1], left = True), [1])))]
        b_lb += [0.]
    
    # i = len(ai_l)-1 case
    A_lb += [list(np.concatenate((-np.array([ai_s[j]*F_r(w, q_l, q_h, ai_l[len(ai_l)-1], ai_s[j], left = True) for j in range(len(ai_s))])/opt_F_r(w, q_l, q_h, w, left = True), [1])))]
    b_lb += [0.]
    
    for i in range(len(ai_r)-1):
        A_lb += [list(np.concatenate((-np.array([ai_s[j]*F_r(w, q_l, q_h, ai_r[i], ai_s[j], left = False) for j in range(len(ai_s))])/opt_F_r(w, q_l, q_h, ai_r[i+1], left = False), [1])))]
        b_lb += [0.]
    
    # i = len(ai_r)-1 case
    A_lb += [list(np.concatenate((-np.array([ai_s[j]*F_r(w, q_l, q_h, ai_r[len(ai_l)-1], ai_s[j], left = False) for j in range(len(ai_s))])/opt_F_r(w, q_l, q_h, np.inf, left = False), [1])))]
    b_lb += [0.]

    A_lb += [list(np.concatenate((np.ones(N_vars), [0])))]
    b_lb += [1.]

    I = matrix(0.0, (N_vars+1,N_vars+1))
    I[::N_vars+2] = -1.0

    A_lb += np.array(I).tolist()
    b_lb += list(np.zeros(N_vars+1))

    c_lb = list(np.zeros(N_vars+1))
    c_lb[-1] = -1.

    A, b, c = matrix(np.vstack(A_lb)), matrix(b_lb), matrix(c_lb)
    sol=solvers.lp(c, A, b)
        
    return {'bound': -sol['primal objective']*100, 'ai_s': ai_s, 'mech': np.array(sol['x']).reshape(-1)[:-1], 'sol': sol}

def upperbound_LP_regular(w, q_l, q_h, eta = 1e-5, b = 10, N = 50): 
    """Function that returns a dictionary  {bound, ai_s, mech, sol}
    where bound is the obtained bound from the LP solution,
    ai_s is support of the near-optimal randomized mechanism 
    mech is the weights of the near-optimal randomized mechanism and
    sol is the solution object for the upperbound LP in section G against Regular distribtions when q in [q_l, q_h]"""
    def F_r(w, q_l, q_h, r, v, left):
        if left:
            if v <= r:
                return 1
            if v <= w:
                return G_a(0, v, r, 1, w, q_l)
            return 0
        else:
            if v <= r:
                return G_a(0, v, 0, 1, w, q_h)
            return 0

    def opt_F_r(w, q_l, q_h, r, left):
        if left:
            return r
        elif r < np.inf:
            return r*G_a(0, r, 0, 1, w, q_h)
        else:
            # lim_{r -> inf} r*G_a(0, r, 0, 1, w, q_h)
            return w/((1/q_h)-1) 

    ai_l = np.linspace(w*q_l, w-eta, N)
    ai_r = np.linspace(w, b, N)
    ai_s = np.append(ai_l, ai_r)
    
    N_vars = len(ai_s)+1
    A_lb = []
    b_lb = []
    
    for i in range(len(ai_l)):
        weights_list = np.concatenate(([ai_s[j]*F_r(w, q_l, q_h, ai_s[i+1], ai_s[j], left = True) for j in range(i+2)], 
                                 [ai_s[j-1]*F_r(w, q_l, q_h, ai_s[i+1], ai_s[j-1], left = True) for j in range(i+2, len(ai_s))]))
        if i == len(ai_l)-1:
            A_lb += [list(np.concatenate((-weights_list/opt_F_r(w, q_l, q_h, w, left = True), [0, 1])))]
        else:
            A_lb += [list(np.concatenate((-weights_list/opt_F_r(w, q_l, q_h, ai_l[i+1], left = True), [0, 1])))]
        b_lb += [0.]
        
    for i in range(len(ai_r)-1):
        A_lb += [list(np.concatenate((-np.array([ai_s[j]*F_r(w, q_l, q_h, ai_r[i+1], ai_s[j], left = False) for j in range(len(ai_s))])/opt_F_r(w, q_l, q_h, ai_r[i+1], left = False), [0, 1])))]
        b_lb += [0.]
    
    # i = len(ai_l)-1 case
    A_lb += [list(np.concatenate((-np.array([ai_s[j]*F_r(w, q_l, q_h, np.inf, ai_s[j], left = False) for j in range(len(ai_s))])/opt_F_r(w, q_l, q_h, np.inf, left = False), [-opt_F_r(w, q_l, q_h, np.inf, left = False)/opt_F_r(w, q_l, q_h, np.inf, left = False), 1])))]
    b_lb += [0.]
   
    A_lb += [list(np.concatenate((np.ones(N_vars), [0])))]
    b_lb += [1.]

    I = matrix(0.0, (N_vars+1,N_vars+1))
    I[::N_vars+2] = -1.0

    A_lb += np.array(I).tolist()
    b_lb += list(np.zeros(N_vars+1))

    c_lb = list(np.zeros(N_vars+1))
    c_lb[-1] = -1.

    A, b, c = matrix(np.vstack(A_lb)), matrix(b_lb), matrix(c_lb)
    sol=solvers.lp(c, A, b)
    
    return {'bound': -sol['primal objective']*100, 'ai_s': ai_s, 'mech': np.array(sol['x']).reshape(-1)[:-1], 'sol': sol}

def lowerbound_LP_mhr(w, q_l, q_h, eta = 1e-5, N = 50): 
    """Function that returns a dictionary  {bound, ai_s, mech, sol}
    where bound is the obtained bound from the LP solution,
    ai_s is support of the near-optimal randomized mechanism 
    mech is the weights of the near-optimal randomized mechanism and
    sol is the solution object for the lowerbound LP against MHR distribtions when q in [q_l, q_h]"""
    def F_r(w, q_l, q_h, r, v, left):
        if left:
            if v <= r:
                return 1
            if v <= w:
                return G_a(1, v, r, 1, w, q_l)
            return 0
        else:
            if v <= r:
                return G_a(1, v, 0, 1, w, q_h)
            return 0

    def opt_F_r(w, q_l, q_h, r, left):
        if left:
            return r
        else :
            return r*G_a(1, r, 0, 1, w, q_h)
        
    if w/np.log(1/q_h) > w :
        ai_l = np.linspace(w/(1+np.log(1/q_l)), w-eta, N)
        ai_r = np.linspace(w, w/np.log(1/q_h), N)
    else:
        ai_l = np.linspace(w/(1+np.log(1/q_l)), w-eta, 2*N)
        ai_r = np.array([w])
        
    ai_s = np.append(ai_l, ai_r)
    
    N_vars = len(ai_s)
    A_lb = []
    b_lb = []
   
    for i in range(len(ai_l)-1):
        A_lb += [list(np.concatenate((-np.array([ai_s[j]*F_r(w, q_l, q_h, ai_s[i], ai_s[j], left = True) for j in range(len(ai_s))])/opt_F_r(w, q_l, q_h, ai_l[i+1], left = True), [1])))]
        b_lb += [0.]
    
    # i = len(ai_l)-1 case
    A_lb += [list(np.concatenate((-np.array([ai_s[j]*F_r(w, q_l, q_h, ai_l[len(ai_l)-1], ai_s[j], left = True) for j in range(len(ai_s))])/opt_F_r(w, q_l, q_h, w, left = True), [1])))]
    b_lb += [0.]
    

    for i in range(len(ai_r)-1):
        A_lb += [list(np.concatenate((-np.array([ai_s[j]*F_r(w, q_l, q_h, ai_r[i], ai_s[j], left = False) for j in range(len(ai_s))])/opt_F_r(w, q_l, q_h, ai_r[i+1], left = False), [1])))]
        b_lb += [0.]
    
    A_lb += [list(np.concatenate((np.ones(N_vars), [0])))]
    b_lb += [1.]

    I = matrix(0.0, (N_vars+1,N_vars+1))
    I[::N_vars+2] = -1.0

    A_lb += np.array(I).tolist()
    b_lb += list(np.zeros(N_vars+1))

    c_lb = list(np.zeros(N_vars+1))
    c_lb[-1] = -1.

    A, b, c = matrix(np.vstack(A_lb)), matrix(b_lb), matrix(c_lb)
    sol=solvers.lp(c, A, b)
    
    return {'bound': -sol['primal objective']*100, 'ai_s': ai_s, 'mech': np.array(sol['x']).reshape(-1)[:-1], 'sol': sol}

def upperbound_LP_mhr(w, q_l, q_h, eta = 1e-5, N = 50): 
    """Function that returns a dictionary  {bound, ai_s, mech, sol}
    where bound is the obtained bound from the LP solution,
    ai_s is support of the near-optimal randomized mechanism 
    mech is the weights of the near-optimal randomized mechanism and
    sol is the solution object for the upperbound LP in section G against MHR distribtions when q in [q_l, q_h]"""
    def F_r(w, q_l, q_h, r, v, left):
        if left:
            if v <= r:
                return 1
            if v <= w:
                return G_a(1, v, r, 1, w, q_l)
            return 0
        else:
            if v <= r:
                return G_a(1, v, 0, 1, w, q_h)
            return 0

    def opt_F_r(w, q_l, q_h, r, left):
        if left:
            return r
        else :
            return r*G_a(1, r, 0, 1, w, q_h)
        
    if w/np.log(1/q_h) > w :
        ai_l = np.linspace(w/(1+np.log(1/q_l)), w-eta, N)
        ai_r = np.linspace(w, w/np.log(1/q_h), N)
    else:
        ai_l = np.linspace(w/(1+np.log(1/q_l)), w-eta, 2*N)
        ai_r = np.array([w])
        
    ai_s = np.append(ai_l, ai_r)
        
    N_vars = len(ai_s)
    A_lb = []
    b_lb = []
    
    for i in range(len(ai_l)):
        weights_list = np.concatenate(([ai_s[j]*F_r(w, q_l, q_h, ai_s[i+1], ai_s[j], left = True) for j in range(i+2)], 
                                 [ai_s[j-1]*F_r(w, q_l, q_h, ai_s[i+1], ai_s[j-1], left = True) for j in range(i+2, len(ai_s))]))
        if i == len(ai_l)-1:
            A_lb += [list(np.concatenate((-weights_list/opt_F_r(w, q_l, q_h, w, left = True), [1])))]
        else: 
            A_lb += [list(np.concatenate((-weights_list/opt_F_r(w, q_l, q_h, ai_l[i+1], left = True), [1])))]
        b_lb += [0.]
        
    for i in range(len(ai_r)-1):
        A_lb += [list(np.concatenate((-np.array([ai_s[j]*F_r(w, q_l, q_h, ai_r[i+1], ai_s[j], left = False) for j in range(len(ai_s))])/opt_F_r(w, q_l, q_h, ai_r[i+1], left = False), [1])))]
        b_lb += [0.]
    
    A_lb += [list(np.concatenate((np.ones(N_vars), [0])))]
    b_lb += [1.]

    I = matrix(0.0, (N_vars+1,N_vars+1))
    I[::N_vars+2] = -1.0

    A_lb += np.array(I).tolist()
    b_lb += list(np.zeros(N_vars+1))

    c_lb = list(np.zeros(N_vars+1))
    c_lb[-1] = -1.

    A, b, c = matrix(np.vstack(A_lb)), matrix(b_lb), matrix(c_lb)
    sol=solvers.lp(c, A, b)
    
    return {'bound': -sol['primal objective']*100, 'ai_s': ai_s, 'mech': np.array(sol['x']).reshape(-1)[:-1], 'sol': sol}