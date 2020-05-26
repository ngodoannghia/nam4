from collections import OrderedDict 
import copy
'''
a = {"a": 4, "b":  1,  "c" : 3}

print(a)

b = sorted(a.items())

print(b[0][1])

tmp = {1 : [4,5], 2: [2,3], 3: [2,1], 4 : [5,4]}

list = []
for op in tmp:
    list.append(op)

c = sorted(tmp.items(), key = lambda kv: (kv[1][0]/ kv[1][1], kv[0]), reverse = True )
'''
'''
list =[1,2,3]

list.append(4)
list.pop(0)
list.append(5)

print(list)

list = [[1,2], [3,4], [2,3], [10, 3],[8,5], [2,1]]
list1 = sorted(list, key = lambda x: (x[0]/x[1]), reverse = True)

print(list1)

a = {}

a['1'] = 1
a['2'] = 2

b = copy.deepcopy(a)

list = []

list.append(b)
a['1'] = 4
a['2'] = 5
b = copy.deepcopy(a)
list.append(b)

print(list)
'''


list = []
for i in range(5):
    a= {}
    a[i] = i 
    a[2*i] = i*i 
    list.append(a)
print(list)
