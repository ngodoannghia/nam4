from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ortools.sat.python import cp_model
model = cp_model.CpModel()
n = 160
q = [model.NewIntVar(1, n, 'q' + str(i)) for i in range(1, n+1)]
# add contrainst
for i in range(n):
    for j in range(i + 1, n):
        model.Add(q[i] != q[j])
        model.Add(q[i] + i != q[j] + j)
        model.Add(q[i] - i != q[j] - j)

solver = cp_model.CpSolver()
status = solver.Solve(model)
if status == cp_model.FEASIBLE:
    for qi in q:
        print(solver.Value(qi), end=' ')
