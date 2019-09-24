# -*- coding: utf-8 -*-
from utils import *
from manager.functions import *
from artificial_intelligence.ai import AI
from environnements.envConnect import Connect
from neural_network_models.modelBuilder import model
from tensorflow.python.keras.models import load_model

def an_exemple_of_model_training():

    a1 = AI(itermax=50,alpha=0.5,use_subtree=True)
    a2 = AI(itermax=100,alpha=0.5,use_subtree=True)
    a3 = AI(itermax=150,alpha=0.5,use_subtree=True)
    a4 = AI(itermax=225,alpha=0.5,use_subtree=True)
    a5 = AI(itermax=300,alpha=0.5,use_subtree=True)

    x, y, l, e = [], [], [], []
    
    rootenv = Connect()
    
    selfplay(rootenv,2000,a1,a1,x,y,l,e)
    selfplay(rootenv,2000,a1,a2,x,y,l,e)
    selfplay(rootenv,2000,a1,a3,x,y,l,e)
    selfplay(rootenv,2000,a1,a4,x,y,l,e)
    selfplay(rootenv,2000,a1,a5,x,y,l,e)
    selfplay(rootenv,2000,a2,a2,x,y,l,e)
    selfplay(rootenv,2000,a2,a3,x,y,l,e)
    selfplay(rootenv,2000,a2,a4,x,y,l,e)
    selfplay(rootenv,2000,a2,a5,x,y,l,e)
    selfplay(rootenv,2000,a3,a3,x,y,l,e)
    selfplay(rootenv,2000,a3,a4,x,y,l,e)
    selfplay(rootenv,2000,a3,a5,x,y,l,e)
    selfplay(rootenv,2000,a4,a4,x,y,l,e)
    selfplay(rootenv,2000,a4,a5,x,y,l,e)
    selfplay(rootenv,2000,a5,a5,x,y,l,e)

    nx = np.array(x)
    py = np.array(y)[:,:-1]
    vy = np.array(y)[:,-1]

    setdata("data/nx",nx)
    setdata("data/py",py)
    setdata("data/vy",vy)
    
    mod = model("master")
    mod.fit(nx, [py,vy], 64, 2, validation_split=0.2)
    mod.save("neural_network_models/pretrained")
