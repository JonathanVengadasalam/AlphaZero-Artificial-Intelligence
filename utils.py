# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from math import sqrt, log, erf

def central_limite_function(x, m): return sqrt(len(x))*(np.mean(x) - m)/np.std(x)

def standard_normal_distribution(x): return (1.0 + erf(x / sqrt(2.0))) / 2.0

def z_test(x, m, alpha=0.05, unilateral=True):
    proba = standard_normal_distribution(central_limite_function(x,m))
    if unilateral: return (1 - alpha) < proba
    return 0 < (proba - alpha/2)*(proba - 1 + alpha/2)

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
