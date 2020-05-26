from __future__ import print_function
import sys
from ortools.sat.python import cp_model
import time



def main(board_size):
    model = cp_model.CpModel()
    # Creates the variables.
    # The array index is the column, and the value is the row.
    queens = [model.NewIntVar(0, board_size - 1, 'x%i' % i)
                for i in range(board_size)]
    # Creates the constraints.
    # The following sets the constraint that all queens are in different rows.
    model.AddAllDifferent(queens)

    # Note: all queens must be in different columns because the indices of queens are all different.

    # The following sets the constraint that no two queens can be on the same diagonal.
    for i in range(board_size):
        # Note: is not used in the inner loop.
        diag1 = []
        diag2 = []
        for j in range(board_size):
        # Create variable array for queens(j) + j.
            q1 = model.NewIntVar(0, 2 * board_size, 'diag1_%i' % i)
            diag1.append(q1)
            model.Add(q1 == queens[j] + j)
        # Create variable array for queens(j) - j.
            q2 = model.NewIntVar(-board_size, board_size, 'diag2_%i' % i)
            diag2.append(q2)
            model.Add(q2 == queens[j] - j)
        model.AddAllDifferent(diag1)
        model.AddAllDifferent(diag2)
    ### Solve model.
    solver = cp_model.CpSolver()
    #solution_printer = SolutionPrinter(queens)
    #status = solver.SearchForAllSolutions(model, solution_printer)
    status = solver.Solve(model)
    if status == cp_model.FEASIBLE:
        print('ok')

if __name__ == '__main__':
  # By default, solve the 8x8 problem.
  start = time.time()
  board_size = 200
  if len(sys.argv) > 1:
    board_size = int(sys.argv[1])
  main(board_size)
  end = time.time()

  print(end - start)