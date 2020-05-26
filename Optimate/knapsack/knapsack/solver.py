#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from ortools.linear_solver import pywraplp


def MIP(n, W, data):

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

    return output
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    listItem = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        listItem.append([int(parts[0]), float(parts[1])])

    # # a trivial greedy algorithm for filling the knapsack
    # # it takes items in-order until the knapsack is full
    # value = 0
    # weight = 0
    # taken = [0]*len(items)

    # K = [[0 for x in range(capacity+1)] for x in range(item_count+1)]

    # for i in range(item_count+1):
    #     for w in range(capacity+1):
    #         if(i == 0 or w == 0):
    #             K[i][w] = 0
    #         elif (items[i-1].weight <= w):
    #             K[i][w] = max(items[i-1].value + K[i-1][w - items[i-1].weight], K[i-1][w])
    #         else:
    #             K[i][w] = K[i-1][w]
    
    # tem = capacity
    # for i in range(item_count, 0, -1):
    #     if (K[i][tem] != K[i-1][tem]): 
    #         taken[i-1] = 1
    #         tem = tem - items[i-1].weight 
    # value = K[item_count][capacity]
    # # prepare the solution in the specified output format
    # output_data = str(value) + ' ' + str(1) + '\n'
    # output_data += ' '.join(map(str, taken))

    output_data = MIP(item_count, capacity, listItem)

    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

