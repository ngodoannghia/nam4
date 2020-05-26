from ortools.sat.python import cp_model
import time


start = time.time()
def SimpleChessProgram():
    model = cp_model.CpModel()
    R = 302
    row = [model.NewIntVar(0, R-1, 'row' + str(x)) for x in range(R)]
    model.AddAllDifferent(row)
    model.Add(row[0] < R//2)
    model.Add(row[-1] >= R//2)
    

    diag1 = []
    diag2 = []

    for i in range(R):
        q1 = model.NewIntVar(i, R+i, 'diag1_%i' %i)
        diag1.append(q1)
        model.Add(q1 == row[i] + i)
        q2 = model.NewIntVar(-i, -i + R, 'diag2_%i' %i)
        diag2.append(q2)
        model.Add(q2 == row[i] - i)

    model.AddAllDifferent(diag1)
    model.AddAllDifferent(diag2)
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status == cp_model.FEASIBLE:
        print('ok')
        for i in row:
            print(solver.Value(i), end=" ")


SimpleChessProgram()
finish = time.time()
print()
print(finish - start)