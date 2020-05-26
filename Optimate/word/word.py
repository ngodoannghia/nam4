INF = 2147483647

def readFile():
    f = open('data', 'r')
    data = f.read()
    data = data.split()
    return data
def printSolution(p, n, text): 
    k = 0
    if p[n] == 1: 
        k = 1
    else: 
        k = printSolution(p, p[n] - 1, text) + 1
    list = []
    for i in range(p[n],n+1,1):
        list.append(text[i-1])
    print(*list)
    return k 
  
def solveWordWrap (l, n, M, text): 
      

    extras = [[0 for i in range(n + 1)] 
                 for i in range(n + 1)] 
                   

    lc = [[0 for i in range(n + 1)] 
             for i in range(n + 1)] 
               

    c = [0 for i in range(n + 1)] 
      
 
    p = [0 for i in range(n + 1)] 

    for i in range(n + 1): 
        extras[i][i] = M - l[i - 1] 
        for j in range(i + 1, n + 1): 
            extras[i][j] = (extras[i][j - 1] - 
                                    l[j - 1] - 1) 
                                      

    for i in range(n + 1): 
        for j in range(i, n + 1): 
            if extras[i][j] < 0: 
                lc[i][j] = INF; 
            elif j == n and extras[i][j] >= 0: 
                lc[i][j] = 0
            else: 
                lc[i][j] = (extras[i][j] * 
                            extras[i][j]) 
  

    c[0] = 0
    for j in range(1, n + 1): 
        c[j] = INF 
        for i in range(1, j + 1): 
            if (c[i - 1] != INF and 
                lc[i][j] != INF and 
                ((c[i - 1] + lc[i][j]) < c[j])): 
                c[j] = c[i-1] + lc[i][j] 
                p[j] = i 
    printSolution(p, n, text)

text = readFile()
n = len(text)
l = []
for i in range(n):
    l.append(len(text[i]))

M = 38
solveWordWrap(l, n, M, text)


