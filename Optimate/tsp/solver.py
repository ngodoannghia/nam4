#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from random import randint
from ortools.linear_solver import pywraplp
from tsp import TSPOpt
from TspMIP import TSP

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def distance(data):
    distances = {}
    for from_counter, from_node in enumerate(data):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(data):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                distances[from_counter][to_counter] = (float(
                    math.hypot((from_node[0] - to_node[0]),
                    (from_node[1]) - to_node[1])))
    return distances

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    output_data = ''
    lines = input_data.split('\n')

    N = int(lines[0])

    data = []
    for i in range(1, N+1):
        line = lines[i]
        parts = line.split()
        data.append((float(parts[0]), float(parts[1])))

    distances = distance(data)

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    # solution = range(0, nodeCount)

    # # calculate the length of the tour
    # obj = length(points[solution[-1]], points[solution[0]])
    # for index in range(0, nodeCount-1):
    #     obj += length(points[solution[index]], points[solution[index+1]])

    # # prepare the solution in the specified output format
    # output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    # output_data += ' '.join(map(str, solution))

    tsp = TSPOpt(N, distances)
    tsp.init()
    it = 0
    while it < 1000000:
        tsp.opt()
        it += 1
    count = 0
    s = 0
    L = []
    L.append(s)
    x = tsp.next[s]
    while ((count < N-1)):
        L.append(x)
        x = tsp.next[x]
        count += 1 
        for i in L:
            if x == i:               
                #print('so dinh: ',count, "dinh", x)
                break
        
    output_data += str(tsp.value)
    output_data += ' 0\n'
    for i in L:
        output_data += str(i)
        output_data += ' '

    # tsp = TSP(N, distances)
    # L, obj = tsp.solverDynamicAddSubTourConstraint()

    # output_data += str(obj)
    # output_data += " 0\n"

    # for i in L:
    #     output_data += str(i)
    #     output_data += ' '

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

