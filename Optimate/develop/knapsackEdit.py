import copy


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

def dictToList(data):
    list = []
    for op in data:
        list.append(op)
    return list
def sortDecense(data):
    return sorted(data, key = lambda kv: (kv[0]/kv[1]), reverse = True)

def bound(node, n, W, data, idx):

    if node[idx]["weight"] >= W:
        return 0
    profit_bound = node[idx]["profit"]
    j = node[idx]["level"] + 1
    totalweight = node[idx]["weight"]

    while (j < n):
        if (totalweight + data[j][1] <= W):
            totalweight += data[j][1]
            profit_bound += data[j][0]
        j += 1
    
    if j < n:
        profit_bound += int((W-totalweight)* (data[j][0] / data[j][1]))
    
    return profit_bound

def knapsack(W, data, n):
    dataSorted = sortDecense(data)
    idx = 0
    queue = []
    listNode = dict()
    parents = {}
    idFinal = -2

    u = {}
    v = {}
    tmp = {}

    u[idx] = {}
    u[idx]["level"] = -1
    u[idx]["profit"] = u[idx]["weight"] = 0
    u["index"] = idx
    
    queue.append(u)
    listNode.update(u)
    idx += 1
    maxProfit = 0
    step = 0

    
    if n <= 200:
        maxStep = 200000
    elif (n >200) & (n <= 1000):
        maxStep = 500000
    else:
        maxStep = 1000000

    while (len(queue) > 0) & (step < maxStep):
      #  print(step)
        step += 1
        u = queue.pop(0)

        temIdx = u["index"]
        v.clear()
        v[idx] = {}

        if(u[temIdx]["level"] == -1):
            v[idx]["level"] = 0
        
        if u[temIdx]["level"] == n-1:
            continue
        
        v[idx]["level"] = u[temIdx]["level"] + 1

        v[idx]["weight"] = u[temIdx]["weight"] + dataSorted[v[idx]["level"]][1]

        v[idx]["profit"] = u[temIdx]["profit"] + dataSorted[v[idx]["level"]][0]
        v["index"] = idx

        v[idx]["bound"] = bound(v, n, W, dataSorted, idx)
        if (v[idx]["weight"] <= W) & (v[idx]["profit"] > maxProfit):
            idFinal = idx
            maxProfit = v[idx]["profit"]
        if ((v[idx]["bound"] < maxProfit) & (v[idx]["profit"] == maxProfit)):
            parents[idx] = temIdx
            tmp = copy.deepcopy(v)
            queue.append(tmp)
            listNode.update(tmp)     
            idx += 1
        elif v[idx]["bound"] > maxProfit:
            parents[idx] = temIdx
            tmp = copy.deepcopy(v)
            queue.append(tmp)
            listNode.update(tmp)     
            idx += 1
        v.clear()
        v[idx] = {}
        v[idx]["weight"] = u[temIdx]["weight"]
        v[idx]["profit"] = u[temIdx]["profit"]
        v[idx]["level"] = u[temIdx]["level"] + 1
        v[idx]["bound"] = bound(v, n, W, dataSorted, idx)
        v["index"] = idx

        if v[idx]["bound"] > maxProfit:
            parents[idx] = temIdx
            tmp = copy.deepcopy(v)
            queue.append(tmp)
            listNode.update(tmp)
            idx += 1

    return maxProfit, idFinal, listNode, parents

def trade(maxProfit, idFinal, listNode, parents, data):
    dataSorted = sortDecense(data)
    result = []
    taken = [0]*len(data)
    idx = idFinal

    while idx:
        preprofit = listNode[parents[idx]]["profit"]
        nodeprofit = listNode[idx]["profit"]
        if maxProfit == dataSorted[listNode[idx]["level"]]:
            result.append(dataSorted[listNode[idx]["level"]])
            break
        if ((maxProfit - preprofit) == dataSorted[listNode[idx]["level"]][0]):
            result.append(dataSorted[listNode[idx]["level"]])
            maxProfit = preprofit
        idx = parents[idx] 

    for item in result:
        for i in range(len(data)):
            if item == data[i]:
                taken[i] = 1
                break

    return taken

 

def main():
    
    f = open('ks_50_0', 'r')
    input_data = f.read()

    n, W, data = readData(input_data)


    maxProfit, idFinal, listNode, parents = knapsack(W, data,n)

    print(parents)
    print(listNode)
    #taken = trade(maxProfit, idFinal, listNode, parents, data)

    #print(taken)

    print(maxProfit, idFinal)

if __name__ == '__main__':
    main()
    