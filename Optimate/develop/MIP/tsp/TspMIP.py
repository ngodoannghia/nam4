from ortools.linear_solver import pywraplp
import numpy as np
import math

class TSP:
    
    def __init__(self, N, distances):
    
        self.N = N
        self.solver = pywraplp.Solver("TSP", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        self.X = {}
        for u in range(self.N):
            self.X[u] = {}
            for v in range(N):
                if u != v:
                    self.X[u][v] =  self.solver.IntVar(0, 1, 'X[%i][%i]' % (u, v))
        self.distances = distances

    
    def findNext(self, s):
        for i in range(self.N):
            if (i != s): 
                if (self.X[s][i].solution_value() > 0):
                    return i 
        return -1
    
    def extractCycle(self, s):
        L = []
        x = s 
        while True:
            L.append(x)
            x = self.findNext(x)
            rep = -1
            for i in range(len(L)):
                if L[i] == x:
                    rep = i 
                    break
            if rep != -1:
                rL = []
                for i in range(len(L)):
                    rL.append(L[i])
                return rL
    def createObjective(self):
        obj = self.solver.Objective()
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    obj.SetCoefficient(self.X[i][j], self.distances[i][j])
    def createFlowConstraint(self):
        for i in range(self.N):
            c1 = self.solver.RowConstraint(1, 1, '')
            c2 = self.solver.RowConstraint(1,1, '')
            for j in range(self.N):
                if j != i:
                    c1.SetCoefficient(self.X[i][j], 1)
                    c2.SetCoefficient(self.X[j][i], 1)
    
    def createSEC(self, S):
        for C in S:
            sc = self.solver.RowConstraint(0, len(C) - 1)
            for i in C:
                for j in C:
                    if i != j:
                        sc.SetCoefficient(self.X[i][j], 1)
 
    def createSolverWithSubTourConstraints(self,S):
        self.createObjective()
        self.createFlowConstraint()
        self.createSEC(S)
    def solverDynamicAddSubTourConstraint(self):
        S = []
        mark = np.empty(self.N, dtype = bool)
        #print(mark)
        found = False

        self.createSolverWithSubTourConstraints(S)

        while(not found):
            self.createSEC(S)

            status = self.solver.Solve()

            if status != pywraplp.Solver.OPTIMAL:
                print("not optimal solution")
                return
            for i in range(self.N):
                mark[i] = False
            S = []
            for s in range(self.N):
                if not mark[s]:
                    C = self.extractCycle(s)
                    if len(C) < self.N:
                        S.append(C)
                        for i in C:
                            mark[i] = True
                    else:
                       # print("Solution not found!!!")
                        found = True 
                        break 
        tour = self.extractCycle(0)
        for i in range(len(tour)):
            print(tour[i], " --> ", end = " ")
        print(tour[0])
        print("objective: ", self.solver.Objective().Value())


def main():
   
    f = open('../data/tsp_200_2','r')
    input_data = f.read()
    lines = input_data.split('\n')

    N = int(lines[0])
    data = []
    for i in range(1, N+1):
        line = lines[i]
        parts = line.split()
        data.append((float(parts[0]), float(parts[1])))
    
    distances = {}
    for from_counter, from_node in enumerate(data):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(data):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                distances[from_counter][to_counter] = (float(
                    math.hypot((from_node[0] - to_node[0]),
                    (from_node[1]) - to_node[1])))
 

    tsp = TSP(N, distances)
    tsp.solverDynamicAddSubTourConstraint()

    #print(N, data)

if __name__=='__main__':
    main()
    


    
