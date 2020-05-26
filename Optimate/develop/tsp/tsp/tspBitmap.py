import math
from collections import namedtuple
import numpy as np
from bitmap import BitMap


Point = namedtuple("Point", ['x', 'y'])
INF = 1000000000


def min(a, b):
    if a > b:
        return b
    else:
        return a

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def slove(input_data):

    
    lines = input_data.split('\n')

    n = int(lines[0])

    trade = []
    tem = 0

    mem = np.full((n,pow(2,n)), -1, dtype='int')
    S = BitMap(n)    

    points = []
    for i in range(1, n+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))
        
    def tsp(i, S):
        
  
        if (S.count() == n):
            return length(points[i], points[0])

        if (mem[i][int(S.tostring(), 2)] != -1):
            return mem[i][int(S.tostring(), 2)]
        res = INF
        for j in range(n):
            print(j)
            if (S.test(j)):
                continue        
            S.set(j)
            oldres = res
            res = min(res, length(points[i], points[j]) + tsp (j, S))

            
        trade.append(tem)
        mem[i][int(S.tostring(), 2)] = res   
        return res

  
    print(tsp(0, S))
    print(trade) 

def main():
    f = open('data/tsp_5_1','r')
    input_data = f.read()
    slove(input_data)

if __name__=='__main__':
    main()
