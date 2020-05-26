import numpy as np 


class ColorGreedy:
    def __init__(self, A, N):
        self.A = A 
        self.N = N 
    
    def solve(self):
        result = {}
        available = {}
        for i in range(self.N):
            result[i] = -1
            available[i] = True
        
        for u in range(self.N):

            for v in self.A[u]:
                if result[v] != -1:
                    available[result[v]] = False
            
            cr = 0
            for i in range(self.N):
                if available[i]:
                    break
                cr += 1
            
            result[u] = cr

            for i in range(self.N):
                available[i] = True
        
        #print(result)
        obj = max(result.values())
        
        #print("Obj: ", obj)

        output = ''
        output += str(obj+1)
        output += ' 0\n'
        for i in range(self.N):
            output += str(result[i])
            output += ' '
        
        return output

# def main():
#     f = open('data/gc_100_5')
#     input_data = f.read()
#     lines = input_data.split('\n')
#     line = lines[0].split()
#     N = int(line[0])
#     E = int(line[1])
#     A = {}
#     for i in range(N):
#         A[i] = []
#     for i in range(1, E+1):
#         line = lines[i].split()
#         u = int(line[0])
#         v = int(line[1])
#         A[u].append(v)
#         A[v].append(u)
    
    # color = Coloring(A, N)
    # color.solve()


# if __name__ == '__main__':
#     main()

        