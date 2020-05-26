import numpy as np
from fractions import Fraction

class simplex:

    def __init__(self,B, b, c, basic, A1):
        self.B = B
        self.b = b
        self.c = c 
        self.basic = basic
        self.A1 = A1
    
    def setValue(self, B, b, c):
        self.B = B
        self.n = b 
        self.c = c

    def setC(self, c):
        self.c = c

    def createtable(self):
        
        self.basic = sorted(self.basic)

        idx = []
        for i in range(len(self.basic)):
            idx.append([self.basic[i][1]])
      
        coffOb = []

        for i in range(len(self.basic)):
            coffOb.append(self.c[self.basic[i][1]])
        
        coffOb = np.transpose([coffOb])


        table = np.hstack((idx, coffOb))

        xb = np.transpose([self.b])

        table = np.hstack((table, xb))

        table = np.hstack((table, self.B))
        table = np.array(table, dtype = 'float')
        # for row in table: 
        #     for el in row: 
        #         print(Fraction(str(el)).limit_denominator(100), end ='\t')  
        #     print() 
        return table

    def checkPhrase1(self):
        if list(self.B.shape)[1] > list(self.A1.shape)[1]:
            return True
        else:
            return False

    def phrase1(self, table):
        sizeB = list(self.B.shape)
        sizeA1 = list(self.A1.shape)
        # for row in table: 
        #     for el in row: 
        #         print(Fraction(str(el)).limit_denominator(100), end ='\t')  
        #     print() 
        reached = 0     
        itr = 1
        unbounded = 0
        alternate = 0
        
        while reached == 0: 
        
            i = 0
            rel_prof = [] 
            while i<len(self.B[0]):
                rel_prof.append(self.c[i] - np.sum(table[:, 1]*table[:, 3 + i])) 
                i = i + 1
            for row in table: 
                for el in row: 
                    print(Fraction(str(el)).limit_denominator(100), end ='\t')  
                print() 
            print("rel profit: ", end =" ") 
            for profit in rel_prof: 
                print(Fraction(str(profit)).limit_denominator(100), end =", ") 
            print() 
            i = 0
            
            b_var = table[:, 0] 

            while i<len(self.B[0]): 
                j = 0
                present = 0
                while j<len(b_var): 
                    if int(b_var[j]) == i: 
                        present = 1
                        break; 
                    j+= 1
                if present == 0: 
                    if rel_prof[i] == 0: 
                        alternate = 1
                        print("Case of Alternate found") 
     
                i+= 1
            print() 
            flag = 0
            for profit in rel_prof: 
                if profit>0: 
                    flag = 1
                    break
  
            if flag == 0: 
                print("All profits are <= 0, optimality reached") 
                reached = 1
                break
        
    
            k = rel_prof.index(max(rel_prof)) 
            min = 99999
            i = 0; 
            r = -1
 
            while i<len(table):
                if (table[:, 2][i]>0 and table[:, 3 + k][i]>0):  
                    val = table[:, 2][i]/table[:, 3 + k][i] 
                    if val<min: 
                        min = val 
                        r = i   
                i+= 1
        
    
            if r ==-1: 
                unbounded = 1
                print("Case of Unbounded") 
                break

            print("pivot element index:", end =' ') 
            print(np.array([r, 3 + k])) 
            pivot = table[r][3 + k] 

                
            table[r, 2:len(table[0])] = table[ 
                    r, 2:len(table[0])] / pivot 
                    

            i = 0
            while i<len(table): 
                if i != r:
                    for j in range(2, len(table[0])):
                        if (j != (k + 3)):
                            table[i][j] = table[i][j] - table[i][3+k]*table[r][j]
                i += 1
            for i in range(len(table)):
                if i != r:
                    table[i][k+3] = 0
        

            table[r][0] = k 
            table[r][1] = self.c[k] 
            
            itr+= 1
        
        if unbounded == 1: 
            print("UNBOUNDED LPP") 
            exit()
        if alternate == 1: 
            print("ALTERNATE Solution") 
    
        i = 0
        while i < len(table):
            if (table[i][0] >= (sizeA1[1])):
                if table[i][2] != 0:
                    print(table)
                    print("Phuong trinh khong co nghiem TM")
                    exit()
                else:
                    table = np.delete(table, (i), axis = 0)
                    i -= 1
            i += 1
        
        F = np.array([table[0]])
        i = 1
        while i < len(table):
            F = np.vstack((F, table[i]))
            i += 1
    
        for i in range(sizeB[0]):
            a = len(F[0]) - 1     
            F = np.delete(F, (a), axis = 1)

        F = np.array(F, dtype='float')

        return F
    
    def phrase2(self, table):

        reached = 0     
        itr = 1
        unbounded = 0
        alternate = 0
        for row in table: 
            for el in row: 
                print(Fraction(str(el)).limit_denominator(100), end ='\t')  
            print() 
        while ((reached == 0) & (itr < 4)): 
            print("vao")
            i = 0
            rel_prof = []

            while (i < (len(table[0]) - 3)): 
                rel_prof.append(self.c[i] - np.sum(table[:, 1]*table[:, 3 + i])) 
                i = i + 1
            for row in table: 
                for el in row: 
                    print(Fraction(str(el)).limit_denominator(100), end ='\t')  
                print() 
            print("rel profit: ", end =" ") 
            for profit in rel_prof: 
                print(Fraction(str(profit)).limit_denominator(100), end =", ") 
            print() 
            i = 0
            
            b_var = table[:, 0] 
  
            while (i < (len(table[0]) - 3)): 
                j = 0
                present = 0
                while j<len(b_var): 
                    if int(b_var[j]) == i: 
                        present = 1
                        break; 
                    j+= 1
                if present == 0: 
                    if rel_prof[i] == 0: 
                        alternate = 1
                        print("Case of Alternate found") 

                i+= 1
            print() 
            flag = 0
            for profit in rel_prof: 
                if profit>0: 
                    flag = 1
                    break

            if flag == 0: 
                print("All profits are <= 0, optimality reached") 
                reached = 1
                break

            k = rel_prof.index(max(rel_prof)) 
            min = 99999
            i = 0; 
            r = -1

            while i<len(table): 
                if (table[:, 2][i]>0 and table[:, 3 + k][i]>0):  
                    val = table[:, 2][i]/table[:, 3 + k][i] 
                    if val<min: 
                        min = val 
                        r = i
                i+= 1
        

            if r ==-1: 
                unbounded = 1
                print("Case of Unbounded") 
                break
        
        
            print("pivot element index:", end =' ') 
            print(np.array([r, 3 + k])) 
        
            pivot = table[r][3 + k] 
            print("pivot element: ", end =" ") 
            print(Fraction(pivot).limit_denominator(100)) 

            table[r, 2:len(table[0])] = table[ 
                    r, 2:len(table[0])] / pivot 
                    
            i = 0
            while i<len(table): 
                if i != r: 
                    table[i, 2:len(table[0])] = table[i, 
                        2:len(table[0])] - table[i][3 + k] * \
                        table[r, 2:len(table[0])] 
                i += 1
        

            table[r][0] = k 
            table[r][1] = self.c[k] 
            
            itr+= 1
        
        if unbounded == 1: 
            print("UNBOUNDED LPP") 
            exit()
        if alternate == 1: 
            print("ALTERNATE Solution") 
            print(table)

        for row in table: 
            for el in row: 
                print(Fraction(str(el)).limit_denominator(100), end ='\t')  
            print()  
        return table

