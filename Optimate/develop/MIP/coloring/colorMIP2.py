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
        M = 100000
        infinity = self.solver.infinity()
        for j in range(self.N):
            X[j] = self.solver.IntVar(0, self.N, 'X[%i]' %j)
        
     
        for u in range(self.N):
            
            for v in self.A[u]:
                constraint1 = self.solver.RowConstraint(-infinity, -1 + M, '')
                constraint1.SetCoefficient(X[u], 1)
                constraint1.SetCoefficient(X[v], -1)
                constraint2 = self.solver.RowConstraint(1,infinity, '')
                constraint2.SetCoefficient(X[u], 1)
                constraint2.SetCoefficient(X[v], -1)

        objective = self.solver.Objective()
        for i in range(self.N):
            objective.SetCoefficient(X[i], 1)
        
        objective.SetMinimization()

        status = self.solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            print('Objective value =', self.solver.Objective().Value())

        else:
            print("no solution")

def main():
    f = open('../data/gc_20_1')
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


