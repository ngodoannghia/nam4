from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math


def length(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
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

def create_distance_matrix(data, manager):
    distances = {}
    index_manager = manager

    for from_counter, from_node in enumerate(data['locations']):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(data['locations']):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                distances[from_counter][to_counter] = length(from_node,to_node)
    print(distances)

    def distance_callback(from_index, to_index):
        from_node = index_manager.IndexToNode(from_index)
        to_node = index_manager.IndexToNode(to_index)

        return distances[from_node][to_node]
    return distance_callback

def print_solution(manager, routing, assigment):
    #print('Objective: {}'.format(assigment.ObjectiveValue()))
    output_data = ''
    index = routing.Start(0)
    #plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    '''
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = assigment.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    plan_output += 'Distance of the route: {}m\n'.format(route_distance)
    print(plan_output)
    '''
    trade = []
    while not routing.IsEnd(index):
        trade.append(index)
        previous_index = index
        index = assigment.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    
    print(route_distance, 1)
  #  output_data += route_distance.tostring()
    for i in range(len(trade)):
        print(trade[i], end=' ')



def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    # [START data]
    f = open('data/tsp_5_1','r')
    input_data = f.read()
    data = create_data_model(input_data)
    # [END data]

    # Create the routing index manager.
    # [START index_manager]
    manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                           data['num_vehicles'], data['depot'])
    # [END index_manager]

    # Create Routing Model.
    # [START routing_model]
    routing = pywrapcp.RoutingModel(manager)
    # [END routing_model]

    # Create and register a transit callback.
    # [START transit_callback]
    distance_callback = create_distance_matrix(data, manager)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # [END transit_callback]

    # Define cost of each arc.
    # [START arc_cost]
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # [END arc_cost]

    # Setting first solution heuristic.
    # [START parameters]
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC)
    # [END parameters]ccccccc

    # Solve the problem.
    # [START solve]
    assignment = routing.SolveWithParameters(search_parameters)
    # [END solve]

    # Print solution on console.
    # [START print_solution]
    if assignment:
        print_solution(manager, routing, assignment)
    # [END print_solution]


if __name__ == '__main__':
    main()