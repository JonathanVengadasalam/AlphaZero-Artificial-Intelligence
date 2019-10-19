# -*- coding: utf-8 -*-
import numpy as np

Target = 4
Height = 6
Width = 7
InitialPlayerMove = -1
PeriodNumber = 3
Depth = 2*PeriodNumber + 1

class Connect:

    def __init__(self, playerjm=None, validmoves=None, indexes=None, terminated=False, state=None):
        
        self.terminated = terminated

        if type(playerjm) == int:
            self.playerJustMoved = playerjm
            self.state = state
            self.validmoves = validmoves
            self.indexes = indexes
        else:
            self.playerJustMoved = InitialPlayerMove
            self.state = np.zeros((Height,Width,Depth))
            self.state[:,:,-1] = 0.5 + self.playerJustMoved*0.5
            self.indexes = np.ones(Width, dtype=int)*(Height-1)
            self.validmoves = list(range(Width))
    
    def __repr__(self):
        s = "JustPlayed:" + str(self.playerJustMoved) + "\n" + self._tostring() + "\n"
        return s
    
    def clone(self):
        return Connect(playerjm = self.playerJustMoved,\
                   state = self.state.copy(),
                   validmoves = self.validmoves.copy(),\
                   indexes = self.indexes.copy(),\
                   terminated = self.terminated)
    
    def convertmove(self, col):
        res = None
        try:
            col = int(col) - 1
            assert self.indexes[col] > -1
            res = col
        except ValueError: print("Le choix n'est pas un nombre")
        except AssertionError: print("Choix impossible")
        except IndexError: print("Choix impossible")
        return res

    def domove(self, col):
        ind = self.indexes[col]
        self.playerJustMoved = -1*self.playerJustMoved
        self.state[:,:,-1] = 1 - self.state[0,0,-1]
        lig = 0 if self.playerJustMoved == 1 else PeriodNumber
        self.state[:,:,lig+2] = self.state[:,:,lig+1]                 # if PeriodNumber changes you have to change
        self.state[:,:,lig+1] = self.state[:,:,lig]                   # this part of code
        self.state[ind,col,lig] = 1
        self.indexes[col] -= 1
        if self.indexes[col] < 0:
            self.validmoves.remove(col)
        if self._testalignment(ind, col, lig):
            self.validmoves = []
            self.terminated = True
    
    def getmoves(self): return self.validmoves.copy()

    def value(self): return 0.5 + int(self.terminated)/2
    
    def x(self): return self.state.copy()
    
    def y(self, move):
        res = np.zeros(Width+1)
        res[move] = 1
        return res
    
    def _convert(self, x1, x2):
        if x1 == 1: return " X "
        if x2 == 1: return " O "
        return "   "
    
    def _countneighbors(self, ind, col, lig, imax, a, b):
        count = 0
        for i in range(1,imax):
            if self.state[ind+a*i, col+b*i, lig] == 0:
                return count
            count += 1
        return count
    
    def _testalignment(self, ind, col, lig):
        ind1, ind2, col1, col2 = ind+1, Height-ind, col+1, Width-col
        if self._countneighbors(ind, col, lig, ind2, 1, 0) + self._countneighbors(ind, col, lig, ind1, -1, 0) + 1 > Target-1: return True
        if self._countneighbors(ind, col, lig, col2, 0, 1) + self._countneighbors(ind, col, lig, col1, 0, -1) + 1 > Target-1: return True
        if self._countneighbors(ind, col, lig, min(ind2,col2), 1, 1) + self._countneighbors(ind, col, lig, min(ind1,col1),-1,-1) + 1 > Target-1: return True
        if self._countneighbors(ind, col, lig, min(ind2,col1), 1,-1) + self._countneighbors(ind, col, lig, min(ind1,col2),-1, 1) + 1 > Target-1: return True
        return False
    
    def _tostring(self):
        width = Width
        st = self._convert(self.state[0,0,0], self.state[0,0,PeriodNumber])
        line = ""
        for i in range(4*width-1):
            line += "-"
        
        for i in range(1, width):
            st += "|" + self._convert(self.state[0,i,0], self.state[0,i,PeriodNumber])
        
        for i in range(1, Height):
            st += "\n" + line + "\n" + self._convert(self.state[i,0,0], self.state[i,0,PeriodNumber])
            for j in range(1, width):
                st += "|" + self._convert(self.state[i,j,0], self.state[i,j,PeriodNumber])
        st += "\n 1 "
        for i in range(1, width):
            st += "  " + str(i+1) + " "
        return st
