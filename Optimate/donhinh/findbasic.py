import numpy as np

class basic:

    def find(self, A):
        b = []
        for j in range(len(A[0])):
            r = -1
            count = 0
            for i in range(len(A)):
                if A[i][j] == 0:
                    count += 1
                else:
                    r = i
            if count == len(A) - 1:
                b.append([r, j])
        a = []
        for i in range(len(b) - 1):
            for j in range(i+1,len(b)):
                if(b[i][0] == b[j][0]):
                    a.append(i)
        if len(a) != 0:
            for i in range(len(a)-1, -1, -1):
                b.remove(b[a[i]])
        return b
    def addVariable(self, A, b):
        a = []
      
        for i in range(len(A)):
            flash = False
            for j in range(len(b)):

                if i == b[j][0]:
                    flash = True
            if (not flash):
                a.append(i)
        
        for i in a:
            arr = []
            for j in range(len(A)):
                if j == i:
                    arr.append([1])
                else:
                    arr.append([0])
            arr = np.array(arr)
            A = np.hstack((A, arr))
            b.append([i, len(A[0]) - 1])
        
        return A, b


# A = np.array([[1, 0, 1],
#              [2, 3, 4],
#              [1, 0, 0]])

# ba = basic()
# b = ba.find(A)
# A1, b = ba.addVariable(A, b)
# print(A1)
# print(b)
