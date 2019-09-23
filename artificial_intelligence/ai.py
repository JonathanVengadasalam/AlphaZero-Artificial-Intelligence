# -*- coding: utf-8 -*-
from tensorflow.python.keras.models import Model
from math import sqrt, log
from time import time
import numpy as np
import random

class Node:
    def __init__(self, playerJustMoved, move=None, parent=None, proba=1, untriedMoves=[]):
        self.move = move
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.quotient = 0
        self.proba = proba
        self.untriedMoves = untriedMoves
        self.playerJustMoved = playerJustMoved
        self.isevaluate = False
    
    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(round(self.wins,1)) + "/" + str(self.visits) + " P:" + str(round(self.proba*100,1))\
            + " Q:" + str(round(self.quotient*100,1)) + " U:" + str(self.untriedMoves) + "]"
    def TreeToString(self, indent):
        s = self.IndentString(indent) + self.__repr__()
        if self.childNodes != []:
            for c in self.childNodes:
                if c.visits > 0:
                    s += c.TreeToString(indent+1)
        return s
    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

class AI:
    count = 1
    def __init__(self, itermax=300, alpha=1, name="", model=None, formula="", use_subtree=False, stochastic=False, temperature=0.5, add_data=True):
        self.turn = 0
        self.add_data = add_data
        self.itermax = itermax
        self.alpha = alpha
        self.use_subtree = use_subtree
        self.name = "a" + str(AI.count) if name == "" else name
        if type(model) is Model:
            self.method, self.model, self.alpha = MODIFIED_MCTS, model, model.output_shape[0][1]*alpha
        else:
            self.method, self.model, self.alpha = CLASSIC_MCTS, None, alpha
        if formula == "UCB1":
            self.formula = _UCB1
        else:
            self.formula = _UCB2
        self.selection, self.temperature = (_Stochastic, temperature) if stochastic else (_Competitive, 0)
        AI.count += 1
    
    def __repr__(self):
        res = "I:" + str(self.itermax) + ", A:" + str(self.alpha) + ", F:" + self.formula.__name__ + ", M:" + self.method.__name__ + ", N:" + self.name
        if self.method is MODIFIED_MCTS: res += "_" + self.model.name
        res += ", S:" + self.selection.__name__
        if self.selection is _Stochastic: res += ", T:" + str(round(self.temperature,2))
        return res + ", U:" + str(self.use_subtree) + ", D:" + str(self.add_data)

    def GetNode(self, rootenv, rootnode):
        return self.selection(self.method(rootenv, rootnode, self.itermax, self.alpha, self.formula, self.model), self.temperature)
        
    def UpdateNode(self, node, move):
        if (type(node) is Node) and self.use_subtree: 
            for c in node.childNodes: 
                if c.move == move: return c
        return None

# Upper Confident Bounce Formula
def _UCB1(node, Visits, alpha): return node.quotient + alpha*sqrt(Visits+1)/(node.visits+1)
def _UCB2(node, Visits, alpha): return node.quotient + alpha*node.proba*sqrt(2*log(Visits+1)/(node.visits+1))

# Move Selection Method
def _Competitive(node, temperature): 
    return sorted(node.childNodes, key = lambda c: c.visits)[-1]
def _Stochastic(node, temperature):
    cursor, step, inverse = 0, 0, 1/temperature
    for c in node.childNodes: cursor += c.visits**inverse
    cursor = cursor*random.random()
    for c in node.childNodes:
        step += c.visits**inverse
        if cursor <= step: return c
    return None

# Node Builder Method
def CLASSIC_MCTS(rootenv, rootnode, itermax, alpha, formula, rnn):
    if type(rootnode) is not Node:
        rootnode = Node(playerJustMoved=rootenv.playerJustMoved, untriedMoves=rootenv.GetMoves())
    
    for i in range(itermax):
        node = rootnode
        env = rootenv.Clone()
        
        # Select
        while node.untriedMoves == [] and node.childNodes != []:
            node = sorted(node.childNodes, key = lambda c: formula(c, node.visits, alpha))[-1]
            env.DoMove(node.move)

        # Expand
        if node.untriedMoves != []:
            m = random.choice(node.untriedMoves)
            env.DoMove(m)
            n = Node(playerJustMoved=env.playerJustMoved, move=m, parent=node, untriedMoves=env.GetMoves())
            node.untriedMoves.remove(m)
            node.childNodes.append(n)
            node = n

        # Rollout - this can often be made orders of magnitude quicker using a env.GetRandomMove() function
        moves = env.GetMoves()
        while moves != []:
            env.DoMove(random.choice(moves))
            moves = env.GetMoves()
        
        # Backpropagate
        value, pjm = env.Value(), env.playerJustMoved
        while node != None:
            node.wins += 0.5 + (value - 0.5)*pjm*node.playerJustMoved
            node.visits += 1
            node.quotient = node.wins/node.visits
            node = node.parentNode

    return rootnode
def MODIFIED_MCTS(rootenv, rootnode, itermax, alpha, formula, rnn):
    if type(rootnode) is not Node: 
        rootnode = Node(playerJustMoved=rootenv.playerJustMoved)
    
    if rootnode.childNodes == []:
        moves = rootenv.GetMoves()
        if moves != []:
            predicts = rnn.predict(np.array([rootenv.X()]))
            tab = predicts[0][0,:]
            for m in moves:
                rootnode.childNodes.append(Node(playerJustMoved=-1*rootnode.playerJustMoved, move=m, parent=rootnode, proba=tab[m]))
    i = 0

    while i < itermax:
        node = rootnode
        env = rootenv.Clone()
        
        # Select
        while node.isevaluate:
            node = sorted(node.childNodes, key = lambda c: formula(c, node.visits, alpha))[-1]
            env.DoMove(node.move)

        # Expand & Asses
        childnodes = node.childNodes
        k, value, pjm = 1, env.Value(), env.playerJustMoved
        if childnodes != []:
            xlist, mlist, vlist = [], [], []
            for c in childnodes:
                _env = env.Clone()
                _env.DoMove(c.move)
                xlist.append(_env.X())
                vlist.append(_env.Value())
                mlist.append(_env.GetMoves())
            predicts = rnn.predict(np.array(xlist))

            for j in range(len(childnodes)):
                c = childnodes[j]
                moves = mlist[j]
                if moves != []:
                    tab = predicts[0][j,:]
                    vlist[j] = predicts[1][j,0]
                    for m in moves:
                        c.childNodes.append(Node(playerJustMoved=-1*c.playerJustMoved, move=m, parent=c, proba=tab[m]))
                c.quotient = c.wins = vlist[j]
                c.visits = 1
            k, value, pjm = len(vlist), sum(vlist), -1*pjm
            node.isevaluate = True
        
        # Backpropagate
        while node != None:
            node.wins += 0.5*k + (value - 0.5*k)*pjm*node.playerJustMoved
            node.visits += k
            node.quotient = node.wins/node.visits
            node = node.parentNode
        i += k

    return rootnode
