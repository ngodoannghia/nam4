#!/usr/bin/python
# -*- coding: utf-8 -*-
from colorGreedy import ColorGreedy
from colorMIP import ColorMip

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')
    line = lines[0].split()
    N = int(line[0])
    E = int(line[1])
    A = {}
    for i in range(N):
        A[i] = []
    for i in range(1, E+1):
        line = lines[i].split()
        u = int(line[0])
        v = int(line[1])
        A[u].append(v)
        A[v].append(u)

    # build a trivial solution
    # every node has its own color
    # solution = range(0, node_count)

    # prepare the solution in the specified output format
    # output_data = str(node_count) + ' ' + str(0) + '\n'
    # output_data += ' '.join(map(str, solution))
    # if (N < 60):
    #     color = ColorMip(A, N, 7)
    #     output_data = color.solve()
    # else:
    color = ColorGreedy(A, N)
    output_data = color.solve()
        
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

