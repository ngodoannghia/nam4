import numpy as np 
import pandas as pd 
from sklearn.impute import KNNImputer
from sklearn import tree

def getData(path):
    data = pd.read_csv(path)
    return data 

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test



def convertList(data):
    size = len(data)
    data2list = []
    class2list = []
    for i in range(size):
        tmp = []
        for att in data:
            if att == 'class':
                class2list.append(data[att][i])
            else:
                tmp.append(data[att][i])
        data2list.append(tmp)
    return data2list, class2list
def createSet(listdata):
    dic = set()
    dictory = {}
    for item in listdata:
        for i in item:
            dic.add(i)
    dic.remove('?')
    dic = sorted(dic, key = None, reverse = False)
    for i in range(len(dic)):
        dictory[dic[i]] = i
    return dictory

def convert2number(resdata, dictory):
    result = []
    for item in resdata:
        tmp = []
        for e in item:
            if e != '?':
                tmp.append(dictory[e])
            else:
                tmp.append(np.nan)
        result.append(tmp)
    
    return result

def convert2num(resClass, dictory):
    result = []
    for item in resClass:
        if item != '?':
            result.append(dictory[item])
        else:
            result.append(np.nan)
    return result

def preprocessData(target):
    listResult = []
    imputer = KNNImputer(n_neighbors = 4, weights= "uniform")
    arrResult = imputer.fit_transform(target)

    for val in arrResult:
        listResult.append(list(val))

    return listResult

def main():
    path = "data/mushrooms.csv"
    data = getData(path)

    train_size = int(len(data)*0.8)

    resData, resClass = convertList(data)

    dictory = createSet(resData)
    tarData = convert2number(resData, dictory)


    tarClass = convert2num(resClass, dictory)
    targetData = list(preprocessData(tarData))

    train_data, test_data = train_test_split(targetData, train_size)
    train_class, test_class = train_test_split(tarClass, train_size)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_class)

    result = list(clf.predict(test_data))

    count = 0
    for i in range(len(result)):
        if result[i] == test_class[i]:
            count += 1
    
    print("Accurancy = ", count / len(result) * 100 ,'%')
            
main()