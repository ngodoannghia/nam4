from ortools.sat.python import cp_model

def SimpleChessProgram():
    model = cp_model.CpModel()
    N = 5
    series = [model.NewIntVar(0, N-1, 'series' + str(x)) for x in range(N)]

    
