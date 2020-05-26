from ortools.sat.python import cp_model

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
    f = open('data/ks_50_0', 'r')
    input_data = f.read()
    n, W, data = readData(input_data)

    model = cp_model.CpModel()

    X = {}
    for j in range(n):
        X[j] = model.NewIntVar(0, 1, 'X[%i]' %j)

    model.Add(sum([X[i] * data[i][1]] for i in range(n)) <= W)

    Z = model.NewIntVar(0, 10000000, 'Z')
    for i in range(n):
        Z += X[i]*data[i][0]
    model.Minimize(Z)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print('obj = ', solver.ObjectiveValue())

main()

