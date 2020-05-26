from __future__ import print_function
from ortools.linear_solver import pywraplp
import numpy as np
import math


def length(node1, node2):
    return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
def readData(input_data):
    lines = input_data.split('\n')
    nodeCount = int(lines[0])
    node = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        node.append((float(parts[0]), float(parts[1])))

    return nodeCount, node 
def main():
    f = open('../data/tsp_51_1', 'r')
    input_data = f.read()

    print(input_data)

    solver = pywraplp.Solver("tsp", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    infinity = solver.infinity()

    nodeCount, node = readData(input_data)


    print(nodeCount)

    x = {}
    y = {}

    for a in range(1, nodeCount):
        y[a] = solver.IntVar(0,nodeCount+1, 'y[%i]' % a)
    for a in range(nodeCount):
        x[a] = {}
        for b in range(nodeCount):
            x[a][b] = solver.IntVar(0, 1, 'x[%i][%i]' % (a, b))

    for a in range(nodeCount):
        constraint = solver.RowConstraint(1, 1, '')
        for b in range(nodeCount):
            if a != b:
                constraint.SetCoefficient(x[a][b], 1)
    
    for b in range(nodeCount):
        constraint = solver.RowConstraint(1, 1, '')
        for a in range(nodeCount):
            if a != b:
                constraint.SetCoefficient(x[a][b], 1)
    
    for a in range(1,nodeCount):       
        for b in range(1, nodeCount):
            if a != b:
                constraint = solver.RowConstraint(-nodeCount, infinity, '')
                constraint.SetCoefficient(y[a], 1)
                constraint.SetCoefficient(x[a][b], -(nodeCount+1))
                constraint.SetCoefficient(y[b], -1)
    objective = solver.Objective()
    for a in range(nodeCount):
        for b in range(nodeCount):
            objective.SetCoefficient(x[a][b], length(node[a], node[b]))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print(solver.Objective().Value(), "1")

        for i in range(1, nodeCount):
            print(int(y[i].solution_value()),end = ' ')
        
        print()

main()
