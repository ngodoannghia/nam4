import numpy as np 

N = 20
INF = 1000000000000
C = np.zeros((N,N))
mem = np.zeros((N, 1 << N), -1, dtype='int')


def min(a, b):
    if a < b:
        return a
    else:
        return b

def read():
    N = input()
    for i in range(N):
        for j in range(N):
            C[i][j] = input()
    
    return N, C
def tsp(i, S):
    if (S == ((1 << N)-1)):
        return C[i][0]
    if (mem[i][S] == -1):
        return mem[i][S]
    res = INF
    for j in range(N):
        if (S & (1 << j)):
            continue
        res = min(res, C[i][j] + tsp(j, S|(1 << j)))
    mem[i][S] = res
    return res
def main():
    read()
    print(C)
    
if __name__ == '__main__':
    main()
