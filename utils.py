# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from math import sqrt, log, erf

def brownian(N):
    
    b = np.insert(  np.random.normal(0., 1., int(N))*np.sqrt(1./N),  0, 0)
    return np.cumsum(b), b

def GBM(So, mu, sigma, W, T):
    
    N = W.shape[0]
    t = np.linspace(0.,T,N)
    S = []
    
    for i in range(0,int(N)):
        S.append(  So*np.exp((mu - 0.5 * sigma**2) * t[i] + sigma * W[i])  )
    return S, t

def CLF(S, m, v, N): return (m + S)/(v*sqrt(N))

def cnd(x): return (1.0 + erf(x / sqrt(2.0))) / 2.0

def getdata(path):
    os.path.exists(path)
    fichier = open(path,"rb")
    tmpPickle = pickle.Unpickler(fichier)
    res = tmpPickle.load()
    fichier.close()
    return res

def setdata(path, donnees):
    fichier = open(path,"wb")
    tmpPickle = pickle.Pickler(fichier)
    tmpPickle.dump(donnees)
    fichier.close()
