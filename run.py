# -*- coding: utf-8 -*-
from manager import AIvsAI
from artificial_intelligence.ai import AI
from environnements.envConnect import Connect
from tensorflow.python.keras.models import load_model
from utils import getdata

m = load_model("neural_network_models/m11288_fited")

a1 = AI(use_subtree=True)
a2 = AI(model=m,use_subtree=True)

x, y, l, e = [], [], [], []
AIvsAI(Connect(),10,a1,a2,x,y,l,e,3)
print("ok")
