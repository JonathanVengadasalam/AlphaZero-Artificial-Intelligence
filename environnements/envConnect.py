# -*- coding: utf-8 -*-
import numpy as np

Target = 4
Height = 6
Width = 7
InitialPlayerMove = -1

class Connect:

    def __init__(self, playerjm=None, state0=None, state1=None, state2=None, validmoves=None, indexes=None, aligned=False):
        
        self.isaligned = aligned

        if type(playerjm) == int:
            self.playerJustMoved = playerjm
            self.state0, self.state1, self.state2 = state0, state1, state2
            self.validmoves = validmoves
            self.indexes = indexes
        else:
            self.playerJustMoved = InitialPlayerMove
            self.state0, self.state1, self.state2 = np.zeros((Height,Width,1)), np.zeros((Height,Width,1)), np.zeros((Height,Width,1))
            self.indexes = np.ones(Width, dtype=int)*(Height-1)
            self.validmoves = list(range(Width))
    
    def __repr__(self):
        s = "JustPlayed:" + str(self.playerJustMoved) + "\n" + self._tostring() + "\n"
        return s
    
    def clone(self):
        return Connect(playerjm = self.playerJustMoved,\
                        state0 = self.state0.copy(), state1 = self.state1.copy(), state2 = self.state2.copy(),
                        validmoves = self.validmoves.copy(),\
                        indexes = self.indexes.copy(),\
                        aligned = self.isaligned)
    
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
        self.state2 = self.state1.copy()
        self.state1 = self.state0.copy()
        self.state0[ind,col,0] = -1*self.playerJustMoved
        
        self.indexes[col] -= 1
        if self.indexes[col] < 0:
            self.validmoves.remove(col)
        if self._isaligned(ind, col):
            self.validmoves = []
            self.isaligned = True
        self.playerJustMoved = -1*self.playerJustMoved
    
    def getmoves(self): return self.validmoves.copy()
    
    #def GetResult(self, playerjm, value): return 0.5 + (value - 0.5)*self.playerJustMoved*playerjm

    def value(self): return 0.5 + int(self.isaligned)/2

    def update_y(self, x, y, value): y[Width] = self.GetResult(x[0,0,3], value)
    
    def x(self): return np.concatenate((self.state0, self.state1, self.state2, np.ones((Height,Width,1))*self.playerJustMoved), -1)
    
    def y(self, move):
        res = np.zeros(Width+1)
        res[move] = 1
        return res
    
    def _convert(self, x):
        if x == 1: return " X "
        if x == -1: return " O "
        if x == 0: return "   "
    
    def _countneighbors(self, ind, col, imax, a, b):
        again, count, i = True, 0, 1
        
        while again and (i < imax):
            if self.state0[ind,col,0] == self.state0[ind+a*i,col+b*i,0]:
                count += 1
            else:
                again = False
            i += 1
        return count
    
    def _isaligned(self, ind, col):
        ind1, ind2, col1, col2 = ind+1, Height-ind, col+1, Width-col
        if self._countneighbors(ind, col, ind2, 1, 0) + self._countneighbors(ind, col, ind1, -1, 0) + 1 > Target-1: return True
        if self._countneighbors(ind, col, col2, 0, 1) + self._countneighbors(ind, col, col1, 0, -1) + 1 > Target-1: return True
        if self._countneighbors(ind, col, min(col2,ind2), 1, 1) + self._countneighbors(ind, col, min(col1,ind1), -1, -1) + 1 > Target-1: return True
        if self._countneighbors(ind, col, min(col1,ind2), 1, -1) + self._countneighbors(ind, col, min(col2,ind1), -1, 1) + 1 > Target-1: return True
        return False
    
    def _tostring(self):
        width = Width
        st = self._convert(self.state0[0,0,0])
        line = ""
        for i in range(4*width-1):
            line += "-"
        
        for i in range(1, width):
            st += "|" + self._convert(self.state0[0,i,0])
        
        for i in range(1, Height):
            st += "\n" + line + "\n" + self._convert(self.state0[i,0,0])
            for j in range(1, width):
                st += "|" + self._convert(self.state0[i,j,0])
        st += "\n 1 "
        for i in range(1, width):
            st += "  " + str(i+1) + " "
        return st
