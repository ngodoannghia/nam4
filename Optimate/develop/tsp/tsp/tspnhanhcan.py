import math
from collections import namedtuple
import numpy as np


Point = namedtuple("Point", ['x', 'y'])
INF = 10000000
f_value = 0
f_min = INF


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def costMin(points, n):
    cmin = INF
    for i in range(n-1):
        for j in range(i+1,n):
            len = length(points[i], points[j])
            if len < cmin:
                cmin = len
    return cmin

def solve(input_data):
    lines = input_data.split('\n')

    n = int(lines[0])
    
    marked = np.zeros(n)
   # a = np.zeros(n)
    a = [-1 for x in range(n+1)]
    trade = [-1 for x in range(n)]

    points = []
    for i in range(1, n+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))
    
    cmin = costMin(points, n)


    def TRY(k):
        global f_value
        global f_min
        for v in range(n):
            if(marked[v] == 0):
                a[k] = v
                f_value = f_value +  length(points[a[k-1]], points[a[k]])
                marked[v] = 1
                if(k==n-1):
                    if (f_value + length(points[a[n-1]], points[0]) < f_min):
                        f_min = f_value+length(points[a[n-1]], points[a[0]])
                        for i in range(n):
                            trade[i] = a[i]
                else:
                    g = f_value + cmin*(n-k+1)
                    if f_value < f_min:
                        TRY(k+1)
                marked[v] = 0
                f_value = f_value - length(points[a[k-1]], points[a[k]])
  
    TRY(0)
    print(f_min)
    print(trade)


def main():
    red = open('data/tsp_5_1','r')
    input_data = red.read()
    solve(input_data)

if __name__=='__main__':
    main()          
