from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ortools.sat.python import cp_model

def SimpleSatProgram():
    model = cp_model.CpModel()

    num_vals = 3
    x = model.NewIntVar(0, num_vals -1, 'x')
    y = model.NewIntVar(0, num_vals -1, 'y')
    z = model.NewIntVar(0, num_vals -1, 'z')

    model.Add(x!=y)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.FEASIBLE:
        print('x = %i' % solver.Value(x))
        print('y = %i' % solver.Value(y))
        print('z = %i' % solver.Value(z))
SimpleSatProgram()