from ortools.sat.python import cp_model

class Coloring:
    def __init__(self, N, A):
        self.N = N
        self.A = A
        self.model = cp_model.CpModel()
    
    def solve(self):
        X = {}
        for j in range(self.N):
            X[j] = self.model.NewIntVar(0, 10, 'X[%i]' % j)

        for u in range(self.N):
            for v in self.A[u]:
                self.model.Add(X[u] != X[v])
        
        Y = self.model.NewIntVar(0, 100000, 'Y')
        for i in range(self.N):
            Y += X[i]
        self.model.Minimize(Y)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 300
        status = solver.Solve(self.model)

        if status == cp_model.OPTIMAL:
            for i in range(self.N):
                print(solver.Value(X[i]), end = ' ')
def main():
    f = open('data/gc_50_1')
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
       # A[v].append(u)
    
    color = Coloring(N, A)
    color.solve()


if __name__ == '__main__':
    main()