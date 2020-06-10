from LinearProgram import simplex
import numpy as np
from fractions import Fraction
import collections

class BranchAndBound:
    
    def solve1(self, A, A_sub, b, c, c_sub):

        # Phrase 1
        sim = simplex(A, A_sub, b, c, c_sub)
        table = sim.createtable()
        table = sim.phrase1(table)
        if table is None:
            print("No solution Phrase 1")
            return None, None, None
        for i in range(len(table)):
            table[i][1] = c[int(table[i][0])]

        print()

        # Phrase 2
        print("Phase 2:")
        table = sim.phrase2(table)
        if table is None:
            print("No solution Phrase 2")
            return None, None, None
        ##Ket qua sau 2 pha
        obj = np.sum(table[:, 1]*table[:, 2])
        print("Obj = ", Fraction(obj).limit_denominator(100))
        result = np.zeros((len(table)))
        for i in range (len(table)):
            result[i] = table[i][2]
        if table is None:
            print("******")
        else:
            for i in range (len(result)):
                print("X"+str(i), "=", Fraction(result[i]).limit_denominator(100))


            for i in range(len(table)):
                for j in range (1, len(table[i])):
                    table[i][j] = Fraction(table[i][j]).limit_denominator(100)

        return result, table, obj
    def solve2(self, A, A_sub, b, c, c_sub):
        # Run Phrase Nhanh can
       # Phrase 1
        sim = simplex(A, A_sub, b, c, c_sub)
        table = sim.createtable()
        table = sim.phrase1(table)
        if table is None:
            print("No solution Phrase 1")
            return None, None, None
        for i in range(len(table)):
            table[i][1] = c[int(table[i][0])]

        print()

        # Phrase 2
        print("Phase 2:")
        table = sim.phrase2Min(table)
        if table is None:
            print("No solution Phrase 2")
            return None, None, None
        ##Ket qua sau 2 pha
        obj = np.sum(table[:, 1]*table[:, 2])
        print("Obj = ", Fraction(obj).limit_denominator(100))
        result = np.zeros((len(table)))
        for i in range (len(table)):
            result[i] = table[i][2]
        if table is None:
            print("******")
        else:
            for i in range (len(result)):
                print("X"+str(i), "=", Fraction(result[i]).limit_denominator(100))


            for i in range(len(table)):
                for j in range (1, len(table[i])):
                    table[i][j] = Fraction(table[i][j]).limit_denominator(100)

        return result, table, obj
def isInteger(fraction):
    if (fraction.denominator == 1):
        return True
    else:
        return False

def convert_f(fraction):
    numerator = fraction.numerator
    denominator = fraction.denominator
    return Fraction(numerator%denominator, denominator)

def branchandbound(A, b, c):
    A_sub = np.hstack((A, np.eye(len(A))))
  
    c_sub = np.zeros((len(c)+len(A)))
    for i in range(len(c), len(c) + len(A)):
       c_sub[i] = 1

    sol = BranchAndBound()
    flash = False
    stop = False
    step  = 0
    result, table, obj = sol.solve1(A, A_sub, b, c, c_sub)
    if table is None:
        print("No solution Two Phrase")
        exit()


    while not flash:
        print("Step = ", step)
        step += 1


        flash = True
        list_val = []   
        for i in range(len(result)):
            if (not isInteger(Fraction(result[i]).limit_denominator(100))):
                flash = False
                list_val.append(convert_f(Fraction(result[i]).limit_denominator(100)))
            else:
                list_val.append(0)
        stop = True
        for i in list_val:
            if i > 0:
                stop = False
        print("result: ", result)

        if stop:
            print("Dat duoc nghiem nguyen")
            obj = np.sum(table[:, 1]*table[:, 2])
            print("obj = ", obj)
            result = np.zeros((len(table)))
            for i in range (len(table)):
                result[i] = table[i][2]
            print(result)
            for row in table: 
                for el in row: 
                    print(Fraction(str(el)).limit_denominator(100), end ='\t') 
                print()
            for i in range(len(result)):
                print("X"+str(table[i, 0]), "=", Fraction(result[i]).limit_denominator(100))
            
            break
   
        ###### Nhanh can ########
        # Tim node re nhanh
        print("result: ",result)
        print("list_v: ", list_val)
        index = list_val.index(max(list_val))
        print("index: ",index)
        value = result[index]
        value_up = int(result[index]) + 1
        print("Value_up: ", value_up)
        value_low = int(result[index])
        print("value_low: ", value_low)
        # Tinh objective UP
        obj_up_global = np.sum(table[:, 1]*table[:, 2])

        # Tinh Objective 
        ###Backup
        table_old = []
        for i in table[:, 2]:
            table_old.append(i)
        
        for i in range(len(table)):
            if list_val[i] != 0:
                table[i, 2] = int(result[i])
        obj_low_golbal = np.sum(table[:, 1]*table[:, 2])

        #Store
        for i in range(len(table)):
            table[i, 2] = table_old[i]

        
        A1 = np.copy(A)
        A2 = np.copy(A)
        c1 = np.copy(c)
        c2 = np.copy(c)
        b1 = np.copy(b)
        b2 = np.copy(b)


    
        ###########
        print("###############################")
        print("Node Phai")
        temA1 = []
        a1 = []
        for i in range(len(A1)):
            a1.append([0])
        A1 = np.hstack((A1, a1))
        idx_A1 = table[index, 0]
        for i in range(len(A1[0]) - 1):
            if i == idx_A1:
                temA1.append(1)
            else:
                temA1.append(0)
        temA1.append(-1)  
        A1 = np.vstack((A1, temA1))
        print("A1")
        print(A1)
        b1 = np.hstack((b1, np.array(value_up)))
        A_sub = np.hstack((A1, np.eye(len(A1))))
        c1 = np.hstack((c1, np.array(0)))
        c_sub = np.zeros((len(c1)+len(A1)))
        for i in range(len(c1), len(c1) + len(A1)):
            c_sub[i] = 1
        print("B: ", b1)
        print("C: ", c1)
        print("A_sub:", A_sub)
        result1, table1, obj1 = sol.solve1(A1, A_sub, b1, c1, c_sub)
        if table1 is None:
            print("No solution fesible")
        else:
            for row in table1: 
                for el in row: 
                    print(Fraction(str(el)).limit_denominator(100), end ='\t') 
                print()
        ##########
        ##########
        print("###############################")
        print("Node Trai")
        temA2 = []
        a2 = []
        for i in range(len(A2)):
            a2.append([0])
        A2 = np.hstack((A2, a2))

        idx_A2 = table[index, 0]
        for i in range(len(A2[0]) - 1):
            if i == idx_A2:
                temA2.append(1)
            else:
                temA2.append(0)
        temA2.append(1)  
        A2 = np.vstack((A2, temA2))
        print("A2")
        print(A2)
        b2 = np.hstack((b2, np.array(value_low)))
        A_sub = np.hstack((A2, np.eye(len(A2))))
        c2 = np.hstack((c2, np.array(0)))
        c_sub = np.zeros((len(c2)+len(A2)))
        for i in range(len(c2), len(c2) + len(A2)):
            c_sub[i] = 1
        print(b2)
        print(c2)
        print("A_sub:", A_sub)
        result2, table2, obj2 = sol.solve1(A2, A_sub, b2, c2, c_sub)

        if table2 is None:
            print("No solution Fesible")
        else:
            for row in table2: 
                for el in row: 
                    print(Fraction(str(el)).limit_denominator(100), end ='\t') 
                print()
        ######################
        print(obj1, obj2)
        
        if ((obj1 is None) and (obj2 is None)):
            print("No solution of problem because: Phrase1 and Phrase2 is None")
            exit()
        if obj1 is None:
            A = np.copy(A2)
            b = np.copy(b2)
            c = np.copy(c2)
            table = np.copy(table2)
            result = np.copy(result2)
            del A1, b1, c1, A2, b2, c2
        elif obj2 is None:
            A = np.copy(A1)
            b = np.copy(b1)
            c = np.copy(c1)
            table = np.copy(table1)
            result = np.copy(result1)
            del A1, b1, c1, A2, b2, c2
        elif obj1 > obj2:
            A = np.copy(A1)
            b = np.copy(b1)
            c = np.copy(c1)
            table = np.copy(table1)
            result = np.copy(result1)
            del A1, b1, c1, A2, b2, c2
        else:
            A = np.copy(A2)
            b = np.copy(b2)
            c = np.copy(c2)
            table = np.copy(table2)
            result = np.copy(result2)
            del A1, b1, c1, A2, b2, c2
def readInput():
    print("Nhap ten file")
    fileName = input()
    f = open(fileName, 'r')
    constraint_num = int(f.readline())
    
    A = []
    for i in range (constraint_num):
    	A.append([float(Aij) for Aij in f.readline().split()])
    c = [float(ci) for ci in f.readline().split()]
    b = [float(bi) for bi in f.readline().split()]

    return np.asarray(A), np.asarray(b), np.asarray(c)
def main():

    A, b, c = readInput()

    # A = np.array([[-1, 3, 1, 0, 0],
    #               [7, 1, 0, 1, 0],
    #               [0, 1, 0, 0, 1]])
    # b = np.array([6, 35, 7])
    # c= np.array([7, 9, 0, 0, 0])

    # A = np.array([[8000, 4000, 1, 0],
    #               [15, 30, 0, 1]])
    
    # b = np.array([40000, 200])
    # c = np.array([100, 150, 0, 0])

    # A = np.array([[-1, 1, 1, 0, 0, 0],
    #               [8, 2, 0, 1, 0, 0],
    #               [1, 0, 0, 0, -1, 0],
    #               [0, 1, 0, 0, 0, 1]])
    # b = np.array([2, 17, 2, 0])
    # c = np.array([5.5, -2.1, 0, 0, 0 , 0])

    branchandbound(A, b, c)


main()
