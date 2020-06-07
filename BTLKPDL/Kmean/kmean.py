################################
################################
#    BAI TOAN PHAN LOAI RUOU   #
################################
################################

import numpy as np 
from numpy import unique 
from numpy import where
import pandas as pd 
from sklearn.cluster import KMeans
from matplotlib import pyplot

def getData(path):
    data = pd.read_csv(path)
    return data 

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

def processData(data):
    X = []
    for i in range(len(data)):
        x = []
        for feature in data: 
            x.append(data[feature][i])
        X.append(x)
    
    X = np.array(X)

    return X

def get_distance(x, y):
    
    return np.sqrt(sum((x-y)**2))

def cal_validation(data, centers, label):
    S = []
    for i in range(len(data)):

        a = get_distance(data[i], centers[label[i]])
        b = 10000000
        for j in range(len(centers)):
            if j != label[i]:
                tem = get_distance(data[i], centers[j])
                if tem < b:
                    b = tem
        #print(a, b)
        s = (b - a)/(max(a, b))
        S.append(s)
    return S

def plot_show(label, data, cell, centers):
    clusters = unique(label)
    n = 100
    for cluster in clusters:
        X = []
        Y = []
        # get row indexes for samples with this cluster
        row_ix = where(label == cluster)
        row_ix = np.array(row_ix)
        
        row = []
        for item in row_ix:
            for i in item:
                row.append(i)

        for i in row:
            tem1 = []
      
            for j in range(cell):
 
                    #print(row_ix.shape)
                    #print(data[i][j])
                    tem1.append(data[i][j])
              
            X.append(np.mean(tem1))
            
        
        X = np.asarray(X)
        C = np.mean(centers[cluster])
        # create scatter of these samples
        pyplot.scatter(X, X)
        pyplot.scatter(C, C, c = n)
        n += 10
    pyplot.show()

def main():
    path = 'data/wine_clustering.csv'

    data = getData(path)

    cell = 0
    for i in data:
        cell += 1
 
    train_size = int(len(data) * 0.8)

    train_data, test_data = train_test_split(data, train_size)
 
    X_train = processData(train_data)

    # xu li du lieu test
    X_test = []
    for i in range(len(test_data)):
        x = []
        for feature in test_data: 
            x.append(test_data[feature][i+train_size])
        X_test.append(x)
    
    X_test = np.array(X_test)

    # ap dung thuat toan K-mean de train
    kmeans = KMeans(n_clusters = 3, random_state = 0).fit(X_train)

    label = kmeans.labels_

    centers = kmeans.cluster_centers_

    # Test K-mean
    resul = kmeans.predict(X_test)

    # Tinh do do theo "silhouette method"  ket hop do gan ket cac phan tu trong cum va do tach roi

    S = cal_validation(X_test, centers, resul)

    f = np.mean(S)
    #print(resul)
    print('validation of silhouette method: ', f)

    #plot_show(label, X_train, cell)

    plot_show(resul, X_test, cell, centers)

main()