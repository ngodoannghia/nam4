import cv2
import os
from random import shuffle
from tqdm import tqdm
import numpy as np


TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
IMG_SIZE = 64
LR = 1e-3

dic = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C' ,'D', 'E','F','G', 'H','I','J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z']
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic') 

def label_img(img):
    lable = [0]*len(dic)

    for i in range(len(dic)):
        #print(img)
        if dic[i] == img:
            lable[i] = 1
    return lable

def create_train_data():

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

            training_data.append([np.array(image), np.array(lable)])
    shuffle(training_data)

    np.save('train_data.npy', training_data)

    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        label = label_img(img)

        for image in os.listdir(path):
            tem = os.path.join(path, image)

            image = cv2.imread(tem, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            testing_data.append([np.array(img), np.array(label)])
    
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)

    return testing_data

train_data = create_train_data()
test_data = process_test_data()


import tflearn 
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 

import tensorflow as tf 

tf.reset_default_graph()
convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 128, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = fully_connected(convnet, 1, activation ='relu') 
convnet = dropout(convnet, 0.8) 
  
convnet = fully_connected(convnet, 36, activation ='softmax') 
convnet = regression(convnet, optimizer ='adam', learning_rate = LR, 
      loss ='mean_square', name ='targets') 
  
model = tflearn.DNN(convnet, tensorboard_dir ='log') 

train_size = int(len(train_data) * 0.8)


train = train_data[:train_size]

test_val = train_data[train_size:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
Y = [i[1] for i in train] 
test_x = np.array([i[0] for i in test_val]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
test_y = [i[1] for i in test_val] 

model.fit({'input': X}, {'targets': Y}, n_epoch = 5,  
    validation_set =({'input': test_x}, {'targets': test_y}),  
    snapshot_step = 500, show_metric = True, run_id = MODEL_NAME) 

model.save(MODEL_NAME)

for test in test_data:


    model_out = model.predict_label(test[0])

    print(np.argmax(model_out))
