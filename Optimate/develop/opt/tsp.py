import numpy as np 
import math
from random import randint

class TSPOpt:
    def __init__(self, N, distances):
        self.N = N
        self.distances = distances
        self.next = {}
        self.pre = {}
        self.value = 0
    
    def init(self):
        for i in range(self.N):
            self.next[i] = (i + 1) % self.N
            self.pre[(i+1) % self.N] = i 
           # print(self.distances[i][(i+1)%self.N])
            self.value += self.distances[i][(i+1)%self.N]
        #print(self.next)

    def opt(self):
        # u = randint(0, self.N - 1)
        # v = randint(0, self.N - 1)
        # while True:
        #     if u == v:
        #         v = randint(0, self.N - 1)
        #     elif self.next[u] == v:
        #         v = randint(0, self.N - 1)
        #     elif self.next[v] == u:
        #         v = randint(0, self.N - 1)
        #     else:
        #         break
        while True:
            u = -1
            v = -1
            value_u = 0
            value_v = 0
            minValue = 10000000
            for i in range(self.N):
                for j in range(self.N):
                    if ((i != j) & (i != self.next[j]) & (j != self.next[i])):
                        value_i = -self.distances[self.pre[i]][i] - self.distances[i][self.next[i]] + \
                            self.distances[i][self.next[j]] + self.distances[self.pre[j]][i]
                        value_j = -self.distances[self.pre[j]][j] - self.distances[j][self.next[j]] + \
                            self.distances[self.pre[i]][j] + self.distances[j][self.next[i]] 
                        if (value_i + value_j < 0):
                            if (value_i + value_j < minValue):
                                minValue = value_i + value_j
                                u = i 
                                v = j
                                value_u = value_i
                                value_v = value_j 
            
            if ((u == -1) | (v == -1)):
                break
            #  doi 2 dinh cho nhau
            # set next u, v
            tem = self.next[u]
            self.next[u] = self.next[v]
            self.pre[self.next[v]] = u
            self.next[v] = tem
            self.pre[tem] = v

            #set pre u, v                
            tem = self.pre[u]
            self.pre[u] = self.pre[v]
            self.next[self.pre[v]] = u
            self.pre[v] = tem
            self.next[tem] = v
            #print(value_u+value_v)
            
            self.value += value_u + value_v
        
        #loai bo 1 dinh
        # else:
        #     value = -self.distances[self.pre[u]][u] - self.distances[u][self.next[u]] - \
        #             - self.distances[v][self.next[v]] + \
        #             self.distances[self.pre[u]][self.next[u]] + self.distances[v][u] + \
        #             self.distances[u][self.next[v]]
        #     if (value < 0):
        #         self.next[self.pre[u]] = self.next[u]
        #         self.pre[self.next[u]] = self.pre[u]
        #         self.pre[u] = v  
        #         self.pre[self.next[v]] = u
        #         self.next[u] = self.next[v]
        #         self.next[v] = u
        #         self.value += value
        #loai bo 2 cung:
def main():
    f = open('data/tsp_200_1','r')
    input_data = f.read()
    lines = input_data.split('\n')

    N = int(lines[0])
    data = []
    for i in range(1, N+1):
        line = lines[i]
        parts = line.split()
        data.append((float(parts[0]), float(parts[1])))
    
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

    tsp = TSPOpt(N, distances)
    tsp.init()
 
    tsp.opt()

    
    count = 0
    s = 0
    L = []
    L.append(s)
    x = tsp.next[s]
    while ((count < N)):
        L.append(x)
        x = tsp.next[x]
        count += 1 
        for i in L:
            if x == i:
                
                print('so dinh: ',count, "dinh", x)
                break
        
    print(len(L))
    print(L)
    print(tsp.value)
   # print(tsp.next)


if __name__=='__main__':
    main()