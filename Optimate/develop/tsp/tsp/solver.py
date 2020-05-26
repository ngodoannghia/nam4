#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math


def create_data_model(input_data):
    data = {}
    lines = input_data.split('\n')

    nodeCount = int(lines[0])
    node = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        node.append((float(parts[0]), float(parts[1])))
    data['locations'] = node
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def compute_euclidean(locations):
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                distances[from_counter][to_counter] = (float(
                    math.hypot((from_node[0] - to_node[0]),
                    (from_node[1]) - to_node[1])))


    return distances

def print_solution(manager, routing, solution):
    index = routing.Start(0)
    output_data = ''
    route_distance = 0
    trade = []
    while not routing.IsEnd(index):
        trade.append(index)
        previous_index = index
      #  print(index)
        index = solution.Value(routing.NextVar(index))
    
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

    route_distance += routing.GetArcCostForVehicle(previous_index, 0, 0)

    output_data += str(route_distance)
    output_data += ' '
    output_data += '1\n'
    for i in trade:
        output_data += str(i)
        output_data += ' '
    return output_data


def solve_it(input_data):

    data = create_data_model(input_data)

    manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'],
                                            data['depot'])

    routing = pywrapcp.RoutingModel(manager)

    distances_matrix = compute_euclidean(data['locations'])

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distances_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

 
    if solution:
        return print_solution(manager, routing, solution)


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        solve_it(input_data)
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

