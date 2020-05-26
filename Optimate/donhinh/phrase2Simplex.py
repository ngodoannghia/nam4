from simplex import simplex
import numpy as np
from fractions import Fraction
from gomory_cut import gomorycut
from dualSimplex import dual
from findbasic import basic


def main():
    # A = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0],
    #           [1, 1, 2, 2, 2, 0, 1, 0, 0], 
    #           [1, 1, 0, 0, 0, 0, 0, 1, 0],
    #           [0, 0, 1, 1, 1, 0, 0, 0, 1]])
    # A = np.array([[2,1,1,0,0],
    #                [1,1,1,1,1],
    #                [1,1,2,2,2],
    #                [1,1,0,0,0],
    #                [0,0,1,1,1]])

    # b = np.array([5, 8, 2, 3])            
        
    # c1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

    # c2 = np.array([2,1,1,0,0])

    

    # A = np.array([[3, 2,1, 0],
    #               [0, 1, 0, 1]])
    # b = np.array([12,2])

    # c = np.array([1,1,0,0])

    # A = np.array([[1, 2, 4, 0, 0, 1, 0, 0],
    #               [0, 4, 2, 1, 0, 0, 1, 0],
    #               [0, 3, 0, 0, 1, 0, 0, 1]])
    # A1 = np.array([[1, 2, 4, 0, 0],
    #                [0, 4, 2, 1, 0],
    #                [0, 3, 0, 0, 1]])
    # A1 = np.array([[1, 2, 4, 0, 0],
    #                [0, 4, 2, 1, 0],
    #                [0, 3, 0, 0, 1]])
    # basic = np.array([0, 3, 4])
    # b = np.array([52, 60, 36])
    # c = [0, 0, 0, 0, 0, 1, 1, 1]
    # c1 = [-2, 6, 4, -2, 3]

    # A = np.array([[1, 1, 2, -1, 0, 0],
    #               [0, -1, -7, 3, 1, 0],
    #               [0, 0, -3, 2, 0, 1]])
    # A1 = np.array([[1, 1, 2, -1, 0, 0],
    #               [0, -1, -7, 3, 1, 0],
    #               [0, 0, -3, 2, 0, 1]])
    
    # b = np.array([2, 3, 7])

    # c = np.array([-2, -1, 1, 1, 0, 0 ])

    # basic = np.array([0, 4, 5])

    # A = np.array([[1, 1, 1, -2, 0, 1, 0],
    #               [-1, 0, 0, 1, 1, 0, 0],
    #               [0, 2, 1, -2, 0, 0, 1]])

    # A = np.array([[1, 1, 1, -2, 0],
    #                [-1, 0, 0, 1, 1],
    #                [0, 2, 1, -2, 0]])
    # c = np.array([0, 0, 0, 0, 0, 1, 1])
    # c1 = np.array([6, 3, 2, -3, 0])

    # b = np.array([4, 10, 12])

    # basic = np.array([5, 4, 6])

    # tim bien co so
    
    # A = np.array([[3, 5, 2, 1, 0, 0],
    #               [4, 4, 4, 0, 1, 0],
    #               [2, 4, 5, 0, 0, 1]])
    
    # A = np.array([[2, 20, 4, 1],
    #               [6, 20, 4, 0]])

    # A = np.array([[1, 1, -1, 0, 0],
    #               [2, -1, 0, -1, 0],
    #               [0, 3, 0, 0, 1]])
    
    # b = np.array([15, 20])
    # A = np.array([[2, -1, 2, -1, 1, 0, 0],
    #               [2, -3, 1, -1, 0, 1, 0],
    #               [-1, 1, -2, -1, 0, 0, 1]])

    ba = basic()
    base = ba.find(A)
    A1, base = ba.addVariable(A, base)

    print(A1)
    print(base)
    # a = []
    # for i in range(len(base)):
    #     a.append(b[i][1])
    #b = np.array([4, -5, 12])
    c1 = np.array([2, 20, -10, 0])
    c  = []
   
    for i in range(len(A1[0])):
        if i < len(A[0]):
            c.append(0)
        else: 
            c.append(1)

    print(c)
    # phrase 1


    gomory = gomorycut()
    sim = simplex(A1, b, c, base, A)
    table = sim.createtable()
    print(table)
    if sim.checkPhrase1():
        table = sim.phrase1(table)
    else:
        sim.setC(c1)
        table = sim.createtable()
    
    result = sim.phrase2(table)

    print(result)
    

    # sim.setValue(c1)

    # result = sim.phrase2(result)

    #print(result)
    # step = 0
    # while ((not gomory.check(result)) & (step < 10)):
    #     print("Step :", step)
    #     step += 1
    #     result = gomory.gomory_cuts(result)




    

if __name__ == "__main__":
    main()