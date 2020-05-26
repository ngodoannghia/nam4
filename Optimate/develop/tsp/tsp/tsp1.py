import math
from collections import namedtuple
import numpy as np


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
    trade = []
    n = int(lines[0])

    mem = np.full((n,pow(2,n)), -1, dtype='int', order = 'C')
    

    points = []
    for i in range(1, n+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))
        

    def tsp(i, S):
        
        if (S == ((1 << n) - 1)):
            return length(points[i], points[0])
    
        if (mem[i][S] != -1):
            return mem[i][S]
        res = INF

        for j in range(n):
            if (S & (1 << j)):
                continue    
            res = min(res, length(points[i], points[j]) + tsp(j, S | (1 << j)))
     
        mem[i][S] = res   
        return res
    
    print(tsp(0, 1 << 0))


def main():
    f = open('data/tsp_5_1','r')
    input_data = f.read()
    slove(input_data)

if __name__=='__main__':
    main()
