import numpy as np 
import math
from random import randint
from ortools.linear_solver import pywraplp


def main():
    f = open('data/fl_50_1')
    input_data = f.read()

    # read Data
    setup_cost = {}
    capacity = []
    locationF = []
    demand = []
    locationC = []
    a = {}
    mask = {}
    f_open = {}



    lines = input_data.split('\n')
    parts = lines[0].split()
    N = int(parts[0])
    M = int(parts[1])
    for i in range(1, N+1):
        parts = lines[i].split()
        setup_cost[i-1] = float(parts[0])
        capacity.append(int(parts[1]))
        locationF.append((float(parts[2]), float(parts[3])))
    
    for i in range(N+1, N+M+1):
        parts = lines[i].split()
        demand.append(int(parts[0]))
        locationC.append((float(parts[1]), float(parts[2])))
    
    for i in range(N):
        f_open[i] = 0

    for i in range(M):
        mask[i] = False
    distances = {}
    for from_counter, from_node in enumerate(locationF):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locationC):
            
            distances[from_counter][to_counter] = (float(
                math.hypot((from_node[0] - to_node[0]),
                (from_node[1]) - to_node[1])))

    f_distances = [[((locationF[i][0] - locationF[j][0]) ** 2 + (locationF[i][1] - locationF[j][1]) ** 2) ** 0.5 \
                        for i in range(len(locationF))]  for j in range(len(locationF))]

    f_indice = [[i for i in range(N)] for j in range(N)]

    for i, row in enumerate(f_indice):
        row.sort(key=lambda x: f_distances[i][x])

    setup = sorted(setup_cost.items(), key = lambda kv : kv[1])
    facility = []
    for i in range(len(setup)):
        facility.append(setup[i][0])
    
    print(facility)
    for i in range(len(facility)):
        u = facility[i]
        cap = capacity[u]
     
        while True:
            v = -1
            minValue = 100000000
            for j in range(M):
                if ((distances[u][j] < minValue) & (demand[j] < cap) & (not mask[j])):
                    minValue = distances[u][j]
                    v = j
            if (v == -1):           
                break
            else:
                a[v] = u 
                cap = cap - demand[v]
                mask[v] = True
                f_open[u] = 1

    obj = 0
    for i in range(M):
        obj += distances[a[i]][i]
    for i in range(N):
        obj += f_open[i] * setup_cost[i]
    best_objective = obj
    n_sub_facility = 20
    round_limit = 1000
    for round in range(round_limit):
        sub_facility = np.random.choice(N)
        #print(sub_facility)
        sub_facility = f_indice[sub_facility][:n_sub_facility]

        sub_facility_set = set(sub_facility)

        print(sub_facility_set)

        sub_customers = [i for i in range(M) if a[i] in sub_facility_set]

        print(sub_customers)

        objective_old = 0.0

        for customer in sub_customers:
            objective_old += f_distances[a[customer]][customer]
        
        for i in sub_facility:
            objective_old += f_open[i] * setup_cost[i]

        solver = pywraplp.Solver('SolveIntegerProblem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        sub_assigment = [[solver.IntVar(0.0, 1.0, 'a'+str(i)+','+str(j)) for j in range(len(sub_facility))] for i in range(len(sub_customers))]

        sub_f_open = [solver.IntVar(0.0, 1.1, 'f'+ str(j)) for j in range(len(sub_facility))]

        for i in range(len(sub_customers)):
            solver.Add(sum([sub_assigment[j][i] for j in range(len(sub_facility))]) == 1)

        for i in range(len(sub_customers)):
            for j in range(len(sub_facility)):
                solver.Add(sub_assigment[j][i] <= sub_f_open[j])
        
        for j in range(len(sub_facility)):
            solver.Add(sum([sub_assigment[j][i] * demand[sub_customers[i]] for i in range(len(sub_customers))]) <= capacity[j])
        
        objective = solver.Objective()

        for i in range(N):
            for j in range(M):
                objective.SetCoefficient(sub_assigment[i][j], f_distances[sub_facility[i]][sub_customers[j]])

        for j in range(len(sub_facility)):
            objective.SetCoefficient(sub_f_open[j], setup_cost[j])

        objective.SetMinimization()

        SEC = 1000
        MIN = 60 * SEC
        solver.SetTimeLimit(1 * MIN)
        result_status = solver.Solve() 

        if result_status != solver.OPTIMAL and result_status != solver.FEASIBLE:
            continue

        objective_new = solver.Objective().Value()
        
        assignment_new = []

        for i in range(len(sub_facility)):
            for j in range(len(sub_customers)):
                if sub_assigment[i][j].solution_value() == 1:
                    assignment_new.append(sub_facility[i])
                    break
        
        if objective_old >= objective_new + 1:
            best_objective -= objective_old - objective_new
            for i, j in enumerate(assignment_new):
                a[sub_customers[i]] = j

            best_f_open = [0]*N 
            for index in a:
                best_f_open[index] = 1
            
            best_output = str(best_objective) + ' '+ '0' + '\n' +' '.join([str(assign) for assign in a])
        #print(best_output)
    return best_output


main()
   

