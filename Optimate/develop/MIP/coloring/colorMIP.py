from ortools.linear_solver import pywraplp
import numpy as np 


class ColorMip:
    def __init__(self, A, N, C):
        self.solver = pywraplp.Solver('simple_mip_program',
                         pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        self.A = A 
        self.N = N 
        self.C = C

    def solve(self):
        X = {}
        Y = {}
        infinity = self.solver.infinity()
        for j in range(self.C):
            Y[j] = self.solver.IntVar(0, 1, 'Y[%i]' %j)
        
        for c in range(self.C):
            X[c] = {}
            for j in range(self.N):
                X[c][j] = self.solver.IntVar(0, 1, 'X[%i][%i]' %(c, j))
        
        for j in range(self.N):
            constraint = self.solver.RowConstraint(1, 1, '')
            for c in range(self.C):
                constraint.SetCoefficient(X[c][j], 1)
        
        for u in range(self.N):
            for v in self.A[u]:
                for c in range(self.C):
                    constraint = self.solver.RowConstraint(-infinity,0, '')
                    constraint.SetCoefficient(X[c][u], 1)
                    constraint.SetCoefficient(X[c][v], 1)
                    constraint.SetCoefficient(Y[c], -1)
        objective = self.solver.Objective()
        for i in range(self.C):
            objective.SetCoefficient(Y[i], 1)
        
        objective.SetMinimization()

        status = self.solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            print('Objective value =', self.solver.Objective().Value())

        else:
            print("no solution")

def main():
    f = open('../data/gc_50_1')
    input_data = f.read()
    lines = input_data.split('\n')
    line = lines[0].split()
    N = int(line[0])
    E = int(line[1])
    A = {}
    for i in range(N):
        A[i] = []
    for i in range(1, E+1):
        line = lines[i].split()
        u = int(line[0])
        v = int(line[1])
        A[u].append(v)
        A[v].append(u)
    color = ColorMip(A, N, 10)
    color.solve()

if __name__ == '__main__':
    main()


