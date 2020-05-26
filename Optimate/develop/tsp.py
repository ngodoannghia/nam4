from __future__ import print_function
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model(input_data):
    data = {}
    lines = input_data.split('\n')
    numLocation = int(lines[0])
    node = []
    for i in range(1,numLocation+1):
        part = lines[i].split()
        node.append((float(part[0]), float(part[1])))
    
    data['location'] = node
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

    print(distances)
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

def main():
    f = open('tsp_18512_1','r')
    input_data = f.read()
    data = create_data_model(input_data)

    manager = pywrapcp.RoutingIndexManager(len(data['location']), data['num_vehicles'],
                                            data['depot'])

    routing = pywrapcp.RoutingModel(manager)

    distances_matrix = compute_euclidean(data['location'])

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distances_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

 
    if solution:
        print(print_solution(manager, routing, solution))

if __name__ == '__main__':
    main()