# -*- coding: utf-8 -*-
import os
import numpy as np
from utils import setdata, z_test
from manager.functions import selfplay
from artificial_intelligence.ai import AI
from environnements import envConnect as env
from tensorflow.python.keras.models import load_model
from neural_network_models.modelBuilder import build_model_for_connect_game

def pretraining_for_connect_game(iteration=2000):

    #initialize object and set parameters
    x, y, l, e = [], [], [], []
    rootenv = env.Connect()
    a1 = AI(itermax=50,alpha=0.5,use_subtree=True)
    a2 = AI(itermax=100,alpha=0.5,use_subtree=True)
    a3 = AI(itermax=150,alpha=0.5,use_subtree=True)
    a4 = AI(itermax=225,alpha=0.5,use_subtree=True)
    a5 = AI(itermax=300,alpha=0.5,use_subtree=True)
    nb_ policy = env.Width
    mod = build_model_for_connect_game(name="pretrained", nb_policy, env.Height, env.Width, env.Depth, nb_resnet=8)
    
    #play game to collect data
    selfplay(rootenv,iteration,a1,a1,x,y,l,e)
    selfplay(rootenv,iteration,a1,a2,x,y,l,e)
    selfplay(rootenv,iteration,a1,a3,x,y,l,e)
    selfplay(rootenv,iteration,a1,a4,x,y,l,e)
    selfplay(rootenv,iteration,a1,a5,x,y,l,e)
    selfplay(rootenv,iteration,a2,a2,x,y,l,e)
    selfplay(rootenv,iteration,a2,a3,x,y,l,e)
    selfplay(rootenv,iteration,a2,a4,x,y,l,e)
    selfplay(rootenv,iteration,a2,a5,x,y,l,e)
    selfplay(rootenv,iteration,a3,a3,x,y,l,e)
    selfplay(rootenv,iteration,a3,a4,x,y,l,e)
    selfplay(rootenv,iteration,a3,a5,x,y,l,e)
    selfplay(rootenv,iteration,a4,a4,x,y,l,e)
    selfplay(rootenv,iteration,a4,a5,x,y,l,e)
    selfplay(rootenv,iteration,a5,a5,x,y,l,e)

    #convert and save data
    nx = np.array(x)
    vy = np.array(y)[:,-1]
    py = np.array(y)[:,:-1]
    setdata("data/pretrained/nx",nx)
    setdata("data/pretrained/py",py)
    setdata("data/pretrained/vy",vy)
    
    #train model and save
    mod.fit(nx, [py,vy], 64, 2, validation_split=0.2)
    mod.save("neural_network_models/pretrained")

    return mod

def selftraining_for_connect(name, iteration=4000):

    #initialize object and set parameters
    x, y, l, e = [], [], [], []
    rootenv = env.Connect()
    mod = load_model("neural_network_models/" + name)
    a = AI(itermax=300,alpha=0.3,name=name,model=mod,formula="UCB1",use_subtree=True,stochastic=True,temperature=0.3)

    #play against himself and collect data
    selfplay(rootenv,iteration,a,a,x,y,l,e)

    #convert and save data
    nx = np.array(x)
    vy = np.array(y)[:,-1]
    py = np.array(y)[:,:-1]
    setdata("data/pretrained/" + name + "/nx",nx)
    setdata("data/pretrained/" + name + "/py",py)
    setdata("data/pretrained/" + name + "/vy",vy)

    #train model and return
    mod.fit(nx, [py,vy], 64, 2, validation_split=0.2)
    mod.save("neural_network_models/" + name + "_new")
    
    return mod

def evaluate_network(name1, name2, iteration=200, alpha=0.05):

    #initialize object and set parameters
    x, y, l, e = [], [], [], []
    rootenv = env.Connect()
    mod1 = load_model("neural_network_models/" + name1)
    mod2 = load_model("neural_network_models/" + name2)
    a1 = AI(itermax=300,alpha=0.3,name=name1,model=mod,formula="UCB1",use_subtree=True,stochastic=True,temperature=0.3)
    a2 = AI(itermax=300,alpha=0.3,name=name2,model=mod,formula="UCB1",use_subtree=True,stochastic=True,temperature=0.3)

    #play game and collect results
    selfplay(rootenv,iteration,a1,a2,x,y,l,e)

    #test mod2
    return z_test(x=l,m=0,alpha=alpha,unilateral=True)

def main(name="master"):
    
    new = selftraining_for_connect(name)
    if evaluate_network(name,name+"_new"):
        new.save(name)
        os.remove(name+"_new")
