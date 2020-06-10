from LinearProgram import simplex
import numpy as np
from convertdata import convert
from fractions import Fraction
from ortools.algorithms import pywrapknapsack_solver
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


def branchandcut(A, b, c):
    A_sub = np.hstack((A, np.eye(len(A))))

    #print(A_sub)
    
    c_sub = np.zeros((len(c)+len(A)))
    for i in range(len(c), len(c) + len(A)):
       c_sub[i] = 1

    # Process
    #sim = simplex(A, A_sub, b, c, c_sub)
    sol = BranchAndBound()
    flash = False
    stop = False
    step  = 0
    result, table, obj = sol.solve2(A, A_sub, b, c, c_sub)
    if table is None:
        print("No solution Two Phrase")
        exit()
    
    ###Knapsack


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

        ###### Nhanh va Cat #########
        
        ### Create x*:
        X = {}
        for i in range(len(A)):
            X[i] = 0
        for i in range(len(table)):
            X[table[i, 0]] = 1 - table[i, 2]

        values = []
        for i in X.keys():
            values.append(-X[i])
        flash_opt = False

        for i in range(len(A)):
            solver = pywrapknapsack_solver.KnapsackSolver(
                pywrapknapsack_solver.KnapsackSolver.
                KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')

            weights = []
            tem = []
            for j in range(len(A[0])):
                tem.append(-A[i][j])
            weights.append(tem)
            capacites = [-b[i]]

            solver.Init(values, weights, capacites)
            computed_value = solver.Solve()

            
            print('Total value =', computed_value)
            x_idx = []
            for k in range(len(values)):
                if solver.BestSolutionContains(k):
                    x_idx.append(k)
                    flash_opt = True
            
            if flash_opt:
                print("X: ", x_idx)
                break

        if flash_opt:
            print("************************")
            print("Run branch and cut")
            print("***********************")
            A_cut = np.copy(A)
            b_cut = np.copy(b)
            c_cut = np.copy(c)
            constraint = []
            sum = 0
            for i in range(len(A_cut[0])):
                if i in x_idx:
                    constraint.append(1)
                    sum += 1
                else:
                    constraint.append(0)  
            constraint.append(1)

            a_coff = []
            for i in range(len(A_cut)):
                a_coff.append([0])
            print(a_coff)
            print(A_cut)
            A_cut = np.hstack((A_cut, a_coff))
            A_cut = np.vstack((A_cut, constraint))
            b_cut = np.hstack((b_cut, np.array(sum - 1)))
            c_cut = np.hstack((c_cut, 0))
            
            A_sub = np.hstack((A_cut, np.eye(len(A_cut))))

            #print(A_sub)
            
            c_sub = np.zeros((len(c_cut)+len(A_cut)))
            for i in range(len(c_cut), len(c_cut) + len(A_cut)):
                c_sub[i] = 1
            result_cut, table_cut, obj_cut = sol.solve2(A_cut, A_sub, b_cut, c_cut, c_sub)
        
        if table_cut is None:
            print("End branch and cut ==> no solution fesible")
            print("Constraint not found")     
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
            result1, table1, obj1 = sol.solve2(A1, A_sub, b1, c1, c_sub)
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
            result2, table2, obj2 = sol.solve2(A2, A_sub, b2, c2, c_sub)

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
        else: 
            print("Branch and cut have solution fesible")  
            A = np.copy(A_cut)
            b = np.copy(b_cut)
            c = np.copy(c_cut)
            table = np.copy(table_cut)
            result = np.copy(result_cut)
            del A_cut, b_cut, c_cut, table_cut, result_cut

def main():

    path = 'data/sc_9_0'
    converts = convert(path)

    A, b, c = converts.solve()

    branchandcut(A, b, c)

main()
