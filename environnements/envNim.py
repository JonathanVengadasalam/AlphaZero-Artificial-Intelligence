# -*- coding: utf-8 -*-
import numpy as np

Height = 1
Width = 3
Depth = 15
InitialPlayerMove = -1

class Nim:

    def __init__(self, playerjm=None, state=None, validmoves=None, index=None, count=None, terminated=False):
        
        self.terminated = terminated

        if type(playerjm) == int:
            self.playerJustMoved = playerjm
            self.state = state
            self.validmoves = validmoves
            self.index = index
            self.count = count
        else:
            self.playerJustMoved = InitialPlayerMove
            self.state = np.zeros((Height,Width,Depth))
            self.state[:,:,-1] = 0.5 + self.playerJustMoved*0.5
            self.validmoves = list(range(Width))
            self.index = 0
            self.count = Depth
    
    def __repr__(self):
        s = "JustPlayed:" + str(self.playerJustMoved) + ", Chips:" + str(self.count) + "\n"
        return s
    
    def clone(self):
        return Nim(playerjm = self.playerJustMoved,\
                        state = self.state.copy(),
                        validmoves = self.validmoves.copy(),\
                        index = self.index,\
                        count = self.count,\
                        terminated = self.terminated)
    
    def convertmove(self, col):
        res = None
        try:
            col = int(col) - 1
            assert -1 < col and col < Width
            res = col
        except ValueError: print("Le choix n'est pas un nombre")
        except AssertionError: print("Choix impossible")
        return res

    def domove(self, col):
        self.playerJustMoved = -1*self.playerJustMoved
        self.state[:,:,-1] = 1 - self.state[0,0,-1]
        self.state[0,col,self.index] = 1
        self.index += 1
        self.count = self.count - col - 1
        if self.count - Width < 0:
            self.validmoves = list(range(self.count))
        if self.count == 0:
            self.validmoves = []
            self.terminated = True
    
    def getmoves(self): return self.validmoves.copy()

    def value(self): return 0.5 + int(self.terminated)/2
    
    def x(self): return self.state.copy()
    
    def y(self, move):
        res = np.zeros(Width+1)
        res[move] = 1
        return res
