from simplex import simplex 
import numpy as np

class dual:

    def __init__(self, c):
        self.c = c 

    def dualSimplex(self, table):
        reach = False
        step = 0
        
        while ((not reach) & (step < 5)):

            step += 1
            print("step: ", step)

            flash = True
            for i in range(len(table)):
                if table[i][2] < 0:
                    flash = False
            
            if flash:
                reach = True
                continue

            pivot = [-1, -1]
            min = 100000
            for i in range(len(table)):
                if table[i][2] < min:
                    pivot[0] = i 
                    min = table[i][2]
            max = -100000
            for i in range(3, len(table[0])):
                if table[pivot[0]][i] < 0:
                    if (c[i-3]/table[pivot[0]][i] > max):
                        max = c[i-3]/table[pivot[0]][i]
                        pivot[1] = i
            
            p = table[pivot[0]][pivot[1]]

            print("pivot: ", pivot[0], pivot[1])
        
            table[pivot[0], 2:len(table[0])] = table[pivot[0], 2:len(table[0])] / p
            i = 0
            while i<len(table): 
                if i != pivot[0]:
                    for j in range(2,len(table[0])):
                        if j != pivot[1]:
                            table[i][j] = table[i][j] - table[i][pivot[1]]*table[pivot[0]][j]
                i += 1
            
            for i in range(len(table)):
                if i != pivot[0]:
                    table[i][pivot[1]] = 0
            print(table)
            for i in range(len(self.c)):
                if (i != (pivot[1]-3)): 
                    print(self.c[i], self.c[pivot[1] - 3], table[pivot[0]][i+3])
                    self.c[i] = self.c[i] - self.c[pivot[1] - 3] * table[pivot[0]][i+3]

            self.c[pivot[1] - 3] = 0

            table[pivot[0]][0] = pivot[1] - 3
            table[pivot[0]][1] = self.c[pivot[1] - 3]

            print(table)
            print(self.c)

        return table
        



        