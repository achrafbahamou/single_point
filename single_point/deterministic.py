from scipy.special import lambertw as W
import numpy as np 

def betaq(q, x):
    """Beta_q function as described in the paper"""
    return 1-(W(x).real+1/W(x).real-2)/np.log(1/q)

def rhoq(q):
    """rho_q function as described in the paper"""
    return betaq(q, 1/np.log(1/q))*np.exp(1-np.log(1/q)*betaq(q, 1/np.log(1/q)))*np.log(1/q)

def opt_deterministic_mhr(q):
    """Function that returns (delta, r) 
    where delta is the optimal deterministic deflation parameters and
    r is the associated optimal ratio against MHR distribtions"""
    if q <= 0.528:
        return betaq(q, np.exp(1)/q), betaq(q, np.exp(1)/q)
    if q <= np.exp(-np.exp(-1)):
        return betaq(q, 1/np.log(1/q)), rhoq(q)
    return 1, np.exp(1)*q*np.log(1/q)

def opt_deterministic_regular(q):
    """Function that returns (delta, r) 
    where delta is the optimal deterministic deflation parameters and
    r is the associated optimal ratio against regular distribtions"""
    if q <= 0.25:
        return 2*np.sqrt(q)/(1+np.sqrt(q)), 2*np.sqrt(q)/(1+np.sqrt(q))
    if q <= 0.5:
        return q*(3-4*q)/(1-q), (3-4*q)/(4*(1-q))
    return 1, 1-q