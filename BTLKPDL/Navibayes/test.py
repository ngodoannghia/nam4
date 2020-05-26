import codecs
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#doc ghi file
def createFilename():
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    filenames = {'business': list1, 'entertainment': list2, 'politics': list3, 'sport': list4, 'tech':list5}

    pathBusiness = 'data/bbc/business/'
    pathPolitics = 'data/bbc/politics/'
    pathSport = 'data/bbc/sport/'
    pathTech = 'data/bbc/tech/'
    pathEntertainment = 'data/bbc/entertainment/'
    # business
    for i in range(1,510+1):
        if i < 10:
            path = pathBusiness + '00' + str(i) + '.txt'
            filenames['business'].append(path)
        elif i < 100:
            path = pathBusiness + '0' +str(i) + '.txt'
            filenames['business'].append(path)
        else:
            path = pathBusiness + str(i) + '.txt'
            filenames['business'].append(path)
    #entertainment
    for i in range(1,386+1):
        if i < 10:
            path = pathEntertainment + '00' + str(i) + '.txt'
            filenames['entertainment'].append(path)
        elif i < 100:
            path = pathEntertainment + '0' +str(i) + '.txt'
            filenames['entertainment'].append(path)
        else:
            path = pathEntertainment + str(i) + '.txt'
            filenames['entertainment'].append(path)
    #politics
    for i in range(1,417+1):
        if i < 10:
            path = pathPolitics + '00' + str(i) + '.txt'
            filenames['politics'].append(path)
        elif i < 100:
            path = pathPolitics + '0' +str(i) + '.txt'
            filenames['politics'].append(path)
        else:
            path = pathPolitics + str(i) + '.txt'
            filenames['politics'].append(path)
    #sport
    for i in range(1,511+1):
        if i < 10:
            path = pathSport + '00' + str(i) + '.txt'
            filenames['sport'].append(path)
        elif i < 100:
            path = pathSport + '0' +str(i) + '.txt'
            filenames['sport'].append(path)
        else:
            path = pathSport + str(i) + '.txt'
            filenames['sport'].append(path)
    #tech
    for i in range(1,401+1):
        if i < 10:
            path = pathTech + '00' + str(i) + '.txt'
            filenames['tech'].append(path)
        elif i < 100:
            path = pathTech + '0' +str(i) + '.txt'
            filenames['tech'].append(path)
        else:
            path = pathTech + str(i) + '.txt'
            filenames['tech'].append(path)
    return filenames
def train_test_split(feature, filnames, ampla):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    filenames_train = {'business': list1, 'entertainment': list2, 'politics': list3, 'sport': list4, 'tech':list5}
    list6 = []
    list7 = []
    list8 = []
    list9 = []
    list10 = []
    filenames_test = {'business': list6, 'entertainment': list7, 'politics': list8, 'sport': list9, 'tech':list10}
    for att in feature:
        for i in range(1, len(filenames[att])*ampla):
            filenames_train[att][i] = filenames[att][i]
        for i in range(len(filenames[att])*ampla+1, len(filenames[att])):
            filenames_test[att][i] = filenames[att][i]
    
    return filenames_train, filenames_test

def DataPreprocess(feature, filenames_train):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    data = {'business': list1, 'entertainment': list2, 'politics': list3, 'sport': list4, 'tech':list5}
    all_stopwords = set(stopwords.words('english'))

    for att in feature:
        for path in filenames[att]:
            f = open(path, 'r', encoding = 'unicode_escape')
            text = f.read()
            text = text.lower()
            #remove unwanted character
            text = re.findall('[\w]+', text)
            list = []
            for w in text:
                if w not in all_stopwords:
                   list.append(w)
            data[att].append(list)
       
    f.close()
    return data


# extract feature
def computeTF(wordDict, bagOfWords, lenght):
    tfDict = {}
    for document in bagOfWords:
        for word, count in wordDict.items():
            tfDict[word] = count / float(lenght)   
    return tfDict


def tf_idf(feature, data):
    uniqueWords = set('')
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    numberOfWords = {'business': list1, 'entertainment': list2, 'politics': list3, 'sport': list4, 'tech':list5}
    lenght = {'business': 0, 'entertainment': 0, 'politics': 0, 'sport': 0, 'tech':0}
    tfDict = {}
    for att in feature:
        for document in  data[att]:
            uniqueWords = uniqueWords.union(set(document))
    print(uniqueWords)
    for att in feature:
        numberOfWords[att] = dict.fromkeys(uniqueWords, 0)
        for document in data[att]:
            lenght[att] += len(document)
            for word in document:
                numberOfWords[att][word] +=1
    for att in feature:
        tfDict[att] = computeTF(numberOfWords[att], data[att], lenght[att])

    print(tfDict)

    
filenames = createFilename()
feature = ['business','sport', 'politics', 'tech', 'entertainment']
data = DataPreprocess(feature, filenames)

tf_idf(feature, data)

#print(data['business'][1])
#print(stopwords.words('english'))
#for document in data['business']:
#    print(document)
#    break








    
