from ortools.sat.python import cp_model

def SimpleChessProgram():
    model = cp_model.CpModel()
    R = 80
    row = [model.NewIntVar(0, R, 'row' + str(x)) for x in range(R)]

    for i in range(R):
        for j in range(i+1, R):
            model.Add(row[i] != row[j])
            model.Add(row[i] != (row[j] + (j - i)))
            model.Add(row[i] != (row[j] - (j - i)))
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status == cp_model.FEASIBLE:
        for i in row:
            print(solver.Value(i), end=' ')

SimpleChessProgram()