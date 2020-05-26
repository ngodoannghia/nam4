from __future__ import print_function
from ortools.linear_solver import pywraplp

def readData(input_data):

    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    listItem = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        listItem.append([int(parts[0]), float(parts[1])])

    return item_count, capacity, listItem

def main():
    f = open('../data/ks_50_0', 'r')
    input_data = f.read()
    n, W, data = readData(input_data)

    #print(data)

    output = ""
    solver = pywraplp.Solver("knapsack", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    infinity = solver.infinity()

    x = {}

    for j in range(n):
        x[j] = solver.IntVar(0, 1, 'x[%i]' % j)
    
    constraint = solver.RowConstraint(0, W, '')
    for j in range(n):
        constraint.SetCoefficient(x[j], data[j][1])
    
    objective = solver.Objective()
    for j in range(n):
        objective.SetCoefficient(x[j], data[j][0])
    objective.SetMaximization()

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        output += str(int(solver.Objective().Value()))
        output += " " 
        output += '1'
        output += '\n'
        for i in range(n):
            output += str(int(x[i].solution_value()))
            output += " "

    print(output)
if __name__ == '__main__':
    main()

    