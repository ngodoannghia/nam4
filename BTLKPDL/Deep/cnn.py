import cv2
import os
from random import shuffle
from random import randint as rand
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
IMG_SIZE = 64
LR = 1e-3

dic = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C' ,'D', 'E','F','G', 'H','I','J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z']


def label_img(img):
    for i in range(len(dic)):
        #print(img)
        if dic[i] == img:
            return i
    return -1
### Tao data train
def create_train_data():
    trainX = []
    trainY = []
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):

        lable = label_img(img)
        #print(lable)
        path = os.path.join(TRAIN_DIR, img)

        for image in os.listdir(path):
            #print(image)
            tem = os.path.join(path, image)
            
            image = cv2.imread(tem, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            training_data.append([np.array(image), lable])
    
    shuffle(training_data)

    np.save('train_data.npy', training_data)

    for item in training_data:
        trainX.append(item[0])
        trainY.append(item[1])
    #print(trainY)
    return np.array(trainX), np.array(trainY)
## Tao data test
def process_test_data():
    testX = []
    testY = []
    testing_data = []

    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        label = label_img(img)

        for image in os.listdir(path):
            tem = os.path.join(path, image)

            image = cv2.imread(tem, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            testing_data.append([np.array(image), label])
    
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)

    for item in testing_data:
        testX.append(item[0])
        testY.append(item[1])

    return np.array(testX), np.array(testY)

### Load Data
def load_data():
    trainX, trainY = create_train_data()
    testX, testY = process_test_data()

    trainX = trainX.reshape((trainX.shape[0], 64, 64, 1))
    testX = testX.reshape((testX.shape[0], 64, 64, 1))

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    # print('Train: X = %s, Y = %s' % (trainX.shape, trainY.shape))
    # print('Test: X = %s, Y = %s' % (testX.shape, testY.shape))

    # print(trainY[0])
    return trainX, trainY, testX, testY

def prep_pixels(train, test):

    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm 

def define_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', input_shape = (64, 64, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(36, activation='softmax'))

    # compiler model

    opt = SGD(lr = 0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    f = open('result/result.txt', 'w')
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
    fold_step = 0
    for train_ix, test_ix in kfold.split(dataX):
        fold_step += 1      
        # define model
        model = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print("*************************************")
        print("Step :", fold_step)
        print('Accurancy: %.3f' % (acc * 100.0))
        print("*************************************")
        f.write("*************************************")
        f.write("Step :", fold_step)
        f.write('Accurancy: %3f' % (acc * 100.0))
        f.write("*************************************")
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories

def predict_dynamic(dataX):
    model = define_model()
    trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
    history = model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY), verbose=0)
    print("**************************")
    print("****** Start Test ********")
    while True:
        a = input()
        if a == 'no':
            break
        idx = rand(0, len(dataX) - 1)
        predict = model.predict(dataX[idx])
        print(predict)

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	
    pyplot.show()


# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()


# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_data()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)


run_test_harness()
