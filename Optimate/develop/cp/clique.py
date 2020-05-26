from ortools.sat.python import cp_model
from random import randint

class clique:
    def __init__(self, N, A):
        self.N = N
        self.A = A
        self.model = cp_model.CpModel()
    
    def solve(self, max_step):
        X = {}
        for j in range(self.N):
            X[j] = self.model.NewIntVar(0, self.N, 'X[%i]' % j)
        clique = []
        mask = {}
        for i in range(self.N):
            mask[i] = False

        count = 0
        while count < max_step:
            tmp = []
            u = randint(0, self.N - 1)
            tmp.append(X[u])

            if len(self.A[u]) == 0:
                continue
            idx = randint(0, len(self.A[u]) - 1)
            v = self.A[u][idx]
            tmp.append(X[v])

            if len(self.A[v]) == 0:
                continue
            idx = randint(0, len(self.A[v])-1)
            k = self.A[v][idx]
            tmp.append(X[k])

            if u == k: 
                continue
            print(u, v, k)

            if tuple((u, v, k)) not in clique:
                clique.append(tuple((u, v, k)))
                mask[u] = True
                mask[v] = True
                mask[k] = True
                self.model.AddAllDifferent(tmp)
            count += 1
        for u in range(self.N):
            if not mask[u]:
                print(mask[u])
                for v in self.A[u]:
                    print(end = " ")
                    #self.model.Add(X[u] != X[v])

        Y = self.model.NewIntVar(0, 1000000, 'Y')
        for i in range(self.N):
            Y += X[i]
        self.model.Minimize(Y)

        solver = cp_model.CpSolver()
        #solver.parameters.max_time_in_seconds = 300
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
        A[v].append(u)
    
    color = clique(N, A)
    color.solve(25)


if __name__ == '__main__':
    main()
