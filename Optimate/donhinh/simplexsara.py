import numpy as np
from findbasic import basic

class Simplex:

    def __init__(self, A, b, c):

        # self.M = len(b)
        # self.N = len(c)
        # self.a = np.zeros([self.M+1,self.M + self.N+1], dtype = float)

        # for i in range(self.M):
        #     for j in range(self.N):
        #         self.a[i][j] = A[i][j]
        # for j in range(self.N, self.M+self.N):
        #     self.a[j-self.N][j] = 1.0
        # for j in range(self.N):
        #     self.a[self.M][j] = c[j]
        # for i in range(self.M):
        #     self.a[i][self.M+self.N] = b[i]
        self.A = A
        ba = basic()
        base = ba.find(A)
        self.a, self.base = ba.addVariable(A, base)

        self.M = len(b)
        #self.N = len(c)
        

        
                
        
    
    
    def pivot(self,p, q):
        for i in range(self.M+1):
            for j in range(len(self.a)):
                if ((i != p) & (j != q)):
                    self.a[i][j] -= self.a[p][j] * self.a[i][q] / self.a[p][q]
        for i in range(self.M+1):
            if i != p:
                self.a[i][q] = 0.0
        for j in range(self.M+self.N+1):
            if j != q:
                self.a[p][j] /= self.a[p][q]
        self.a[p][q] = 1.0
    def solve(self):
        while(True):
            p = 0
            q = 0
            print(self.a)
            while (q < self.M + self.N):
                if (self.a[self.M][q] > 0):
                    break
                q += 1
            if (q >= self.M + self.N):
                break

            while (p < self.M):
                if self.a[p][q] > 0:
                    break
                p += 1
            
            for i in range(p + 1, self.M):
                if self.a[i][q] > 0:
                    if (self.a[i][self.M+self.N] / self.a[i][q] < self.a[p][self.M+self.N] / self.a[p][q]):
                        p = i  
            print(p, q)          
            self.pivot(p, q)
        
      

def main():

    A = np.array([[5, 15],[4, 4], [35, 20]])
    b = [480, 160, 1190]
    c = [13, 23]
    ob = Simplex(A, b, c)
    ob.solve()

if __name__ == '__main__':
    main()