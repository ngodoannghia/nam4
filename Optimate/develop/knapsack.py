


def readData(input_data):

    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = {}
    listItem = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items[i] = [int(parts[0]), float(parts[1])]

    return item_count, capacity, items

def dictToList(data):
    list = []
    for op in data:
        list.append(op)
    return list
def sortDict(data):
    list = sorted(data.items(), key = lambda kv: (kv[1][0]/kv[1][1]), reverse = True)
    dict = {}
    for item in list:
        dict[item[0]] = item[1]
    return dict

def bound(node, n, W, data, listKey):

    if node["weight"] >= W:
        return 0
    profit_bound = node["profit"]
    j = node["level"] + 1
    totalweight = node["weight"]


    while j < n:
        print(j)
        if (totalweight + data[listKey[j]][1] <= W):
            totalweight += data[listKey[j]][1]
            profit_bound += data[listKey[j]][0]
            j += 1
        else:
            break
    
    if j < n:
        profit_bound += (W-totalweight)* (data[listKey[j]][0] / data[listKey[j]][1])
    
    return profit_bound

def knapsack(W, data, n):

    dataSorted = sortDict(data)

#    print(dataSorted)

    listKey = []

    for op in dataSorted:
        listKey.append(op)
    queue = []

    u = {}
    v = {}

    u["level"] = -1
    u["profit"] = u["weight"] = 0
    queue.append(u)
    maxProfit = 0
    step = 0
    while (len(queue) > 0):
        step += 1
        u = queue.pop(0)

        if(u["level"] == -1):
            v["level"] = 0
        
        if u["level"] == n-1:
            continue
        
        v["level"] = u["level"] + 1
        
        v["weight"] = u["weight"] + data[listKey[v["level"]]][1]

        v["profit"] = u["profit"] + data[listKey[v["level"]]][0]

        if (v["weight"] <= W) & (v["profit"] > maxProfit):
            maxProfit = v["profit"]
        
  #      print(v['level'],v['weight'], v["profit"])

        v["bound"] = bound(v, n, W, dataSorted, listKey)

        print("v[bound] :",v["bound"])

        if v["bound"] > maxProfit:
            queue.append(v)
   
        v["weight"] = u["weight"]
        v["profit"] = u["profit"]
        v["bound"] = bound(v, n, W, dataSorted, listKey)

        if v["bound"] > maxProfit:
            queue.append(v)

    return maxProfit

def main():
    
    f = open('ks_4_0', 'r')
    input_data = f.read()

    n, W, data = readData(input_data)

    print(knapsack(W, data,n))

if __name__ == '__main__':
    main()
    