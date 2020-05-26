import numpy as np 
import pandas as pd 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def getData(path):
    data = pd.read_csv(path)
    return data 

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

        

def preprocessTrain(cat, text, size_s, size_e):
    
    doc = []
    y = []
    for idx in range(size_s, size_e):
        doc.append(text[idx].lower()) 
    vectorizer =  TfidfVectorizer()
    x = vectorizer.fit_transform(doc).toarray()
    
    for idx in range(size_s, size_e):
        if cat[idx] == "business":
            y.append(0)
        elif cat[idx] == "entertainment":
            y.append(1)
        elif cat[idx] == "politics":
            y.append(2)
        elif cat[idx] == "sport":
            y.append(3)
        else:
            y.append(4)
  
    return x, y, vectorizer.vocabulary_

def preprocessTest(cat, text, size_s, size_e, vocabulary):

    doc = []
    y = []
    for idx in range(size_s, size_e):
        doc.append(text[idx].lower()) 
    vectorizer =  TfidfVectorizer(vocabulary = vocabulary)
    x = vectorizer.fit_transform(doc).toarray()

    for idx in range(size_s, size_e):
        if cat[idx] == "business":
            y.append(0)
        elif cat[idx] == "entertainment":
            y.append(1)
        elif cat[idx] == "politics":
            y.append(2)
        elif cat[idx] == "sport":
            y.append(3)
        else:
            y.append(4)

    return x, y
def main():
    path = "../data/bbc-text.csv"
    data = getData(path)


    train_size = int(len(data)*0.8)

    train_cat, test_cat = train_test_split(data['category'], train_size)
    train_text, test_text = train_test_split(data['text'], train_size)



    x_train, y_train , vocabulary= preprocessTrain(train_cat, train_text,0, train_size)
    x_test, y_test = preprocessTest(test_cat, test_text, train_size, len(data),vocabulary)


    print(x_train.shape)
    print(x_test.shape)
    model = MultinomialNB()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            count += 1
    
    print(count)
    print("Accurancy = ", count/(len(y_test))*100, "%")
 

if __name__ == "__main__":
    main()
