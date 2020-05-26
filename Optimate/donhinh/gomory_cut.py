from fractions import Fraction
import numpy as np

class gomorycut:


    def convert(self, num):
        a = num.numerator
        b = num.denominator
        c = Fraction(a%b,b)
        return c 

    def isInteger(self, fraction):
        if (fraction.denominator == 1):
            return True
        else:
            return False
    
    def check(self, table):

        for i in range(len(table)):
            if not self.isInteger(Fraction(table[i][2]).limit_denominator(100)):
                return False
        
        return True

    def gomory_cuts(self, table):
        max = [-100000, -1]
        for i in range(len(table)):
            a = Fraction(str(table[i][2])).limit_denominator(100)
            if not self.isInteger(a):
                b =self.convert(a)
                if b > max[0]:
                    max[0] = b
                    max[1] = i
        listIdx = []
        for i in range(len(table)):
            listIdx.append(table[i][0])
        
        constraint = []
        constraint.append(len(table[0]) - 3)
        table = np.hstack((table, np.zeros((len(table), 1))))


        constraint.append(0)
        constraint.append(-float(max[0]))

        for i in range(3, len(table[0])-1):
            if i not in listIdx:
                a = self.convert(Fraction(str(table[max[1]][i])).limit_denominator(100))
                constraint.append(-float(a))
            else:
                constraint.append(0)
        constraint.append(1)

        table = np.vstack((table, np.array(constraint)))

        return table


        




        
    




        
        


