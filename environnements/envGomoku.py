# -*- coding: utf-8 -*-
import numpy as np

Target = 5
Height = 8
Width = 8
InitialPlayerMove = -1
PeriodNumber = 5
Depth = 2*PeriodNumber + 1

class Gomoku:

    def __init__(self, playerjm=None, validmoves=None, lastmove=None, terminated=False, state=None):
        
        self.terminated = terminated

        if type(playerjm) == int:
            self.playerJustMoved = playerjm
            self.state = state
            self.validmoves = validmoves
            self.lastmove = lastmove
        else:
            self.playerJustMoved = InitialPlayerMove
            self.state = np.zeros((Height,Width,Depth))
            self.state[:,:,-1] = 0.5 + self.playerJustMoved*0.5
            self.validmoves = list(range(Width*Height))
            self.lastmove = None
    
    def __repr__(self):
        s = "P:" + str(self.playerJustMoved) + "   M:" + str(self.lastmove) + "\n" + self._tostring() + "\n"
        return s
    
    def clone(self):
        return Gomoku(playerjm = self.playerJustMoved,\
                        state = self.state.copy(),
                        validmoves = self.validmoves.copy(),\
                        lastmove = self.lastmove,\
                        terminated = self.terminated)
    
    def convertmove(self, move):
        res = None
        try:
            lig, col = int(move[0]), int(move[1])
            assert -1 < lig and lig < Height and -1 < col and col < Width
            res = lig*Width + col
        except ValueError: print("Le choix n'est pas un nombre")
        except AssertionError: print("Choix impossible")
        except IndexError: print("Choix impossible")
        return res

    def domove(self, move):
        self.playerJustMoved = -1*self.playerJustMoved
        lig, col, dep = move//Width, move%Width, 0 if self.playerJustMoved == 1 else PeriodNumber
        dep2 = dep + PeriodNumber - 1

        self.state[:,:,-1] = 1 - self.state[0,0,-1]
        for i in range(0, PeriodNumber - 1):
            self.state[:,:,dep2 - i] = self.state[:,:,dep2 - i - 1]  
        self.state[lig, col, dep] = 1
        self.lastmove = (lig, col)

        self.validmoves.remove(move)
        if self._testalignment(lig, col, dep):
            self.validmoves = []
            self.terminated = True
    
    def getmoves(self): return self.validmoves.copy()

    def value(self): return 0.5 + int(self.terminated)/2
    
    def x(self): return self.state.copy()
    
    def y(self, move):
        res = np.zeros(Width*Height+1)
        res[move] = 1
        return res

    def _convert(self, x1, x2):
        if x1 == 1: return " X "
        if x2 == 1: return " O "
        return " . "
    
    def _countneighbors(self, lig, col, dep, imax, a, b):
        count = 0
        for i in range(1,imax):
            if self.state[lig+a*i, col+b*i, dep] == 0:
                return count
            count += 1
        return count
    
    def _testalignment(self, lig, col, dep):
        lig1, lig2, col1, col2 = lig+1, Height-lig, col+1, Width-col
        if self._countneighbors(lig, col, dep, lig2, 1, 0) + self._countneighbors(lig, col, dep, lig1, -1, 0) + 1 > Target-1: return True
        if self._countneighbors(lig, col, dep, col2, 0, 1) + self._countneighbors(lig, col, dep, col1, 0, -1) + 1 > Target-1: return True
        if self._countneighbors(lig, col, dep, min(lig2,col2), 1, 1) + self._countneighbors(lig, col, dep, min(lig1,col1),-1,-1) + 1 > Target-1: return True
        if self._countneighbors(lig, col, dep, min(lig2,col1), 1,-1) + self._countneighbors(lig, col, dep, min(lig1,col2),-1, 1) + 1 > Target-1: return True
        return False
    
    def _tostring(self):
        st = "     "
        for i in range(Width):
            st += " " + str(i) + " "
        
        line1, line2 = "     ", "-----"
        for i in range(3*Width):
            line1 += " "
            line2 += "-"

        st += "\n" + line2 + "\n " + str(0) + " | "
        for i in range(Width):
            st += self._convert(self.state[0,i,0], self.state[0,i,PeriodNumber])
        
        for i in range(1, Height):
            st += "\n" + line1 + "\n " + str(i) + " | "
            for j in range(Width):
                st += self._convert(self.state[i,j,0], self.state[i,j,PeriodNumber])
        return st
