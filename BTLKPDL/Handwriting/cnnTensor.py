# from tensorflow.examples.tutorials.mnist import input_data
# data = input_data.read_data_sets('data/MNIST/', one_hot=True)

# x_batch, y_true_batch = data.train.next_batch(100)

# print(x_batch)
# print(x_batch[0])
# print(y_true_batch.shape)
# print(x_batch.shape)
# print(x_batch[0].shape)

# import cv2

# img = cv2.imread('data/train/0/hsf_0_00000.png', 0) 
# img = cv2.resize(img, (28, 28))/255 

# X = img.reshape(-1)

# print(X)
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
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
#MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic') 

def label_img(img):
    lable = [0]*len(dic)

    for i in range(len(dic)):
        #print(img)
        if dic[i] == img:
            lable[i] = 1
    return lable

def load(dir):

    training_data = []
    for img in tqdm(os.listdir(dir)):

        lable = label_img(img)
        #print(lable)
        path = os.path.join(TRAIN_DIR, img)

        for image in os.listdir(path):
            #print(image)
            tem = os.path.join(path, image)
            
            image = cv2.imread(tem, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))/255


            X = image.reshape(-1)
            
            training_data.append([X, lable])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    X = []
    Y = []

    for i in range(len(training_data)):
        X.append(training_data[i][0])
        Y.append(training_data[i][1])

    #print(np.array([X]))

    return np.array(X), np.array(Y)

x = tf.placeholder(tf.float32, shape=[None, 64*64], name='x')
print(x)
x_image = tf.reshape(x, [-1, 64, 64, 1])

y_true = tf.placeholder(tf.float32, shape=[None, 36], name='y_true')
print(y_true)
y_true_cls = tf.argmax(y_true, dimension=1)

def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
    
    with tf.variable_scope(name) as scope:
     
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        layer += biases
        
        return layer, weights

def new_pool_layer(input, name):
    
    with tf.variable_scope(name) as scope:
   
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        return layer

def new_relu_layer(input, name):
    
    with tf.variable_scope(name) as scope:
        layer = tf.nn.relu(input)
        
        return layer

def new_fc_layer(input, num_inputs, num_outputs, name):
    
    with tf.variable_scope(name) as scope:

       
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
        
        layer = tf.matmul(input, weights) + biases
        
        return layer
# Convolutional Layer 1
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=1, filter_size=5, num_filters=6, name ="conv1")

# Pooling Layer 1
layer_pool1 = new_pool_layer(layer_conv1, name="pool1")

# RelU layer 1
layer_relu1 = new_relu_layer(layer_pool1, name="relu1")

# Convolutional Layer 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_relu1, num_input_channels=6, filter_size=5, num_filters=16, name= "conv2")

# Pooling Layer 2
layer_pool2 = new_pool_layer(layer_conv2, name="pool2")

# RelU layer 2
layer_relu2 = new_relu_layer(layer_pool2, name="relu2")

# Flatten Layer
num_features = layer_relu2.get_shape()[1:4].num_elements()
layer_flat = tf.reshape(layer_relu2, [-1, num_features])

# Fully-Connected Layer 1
layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")

# RelU layer 3
layer_relu3 = new_relu_layer(layer_fc1, name="relu3")

# Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=36, name="fc2")

with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

with tf.name_scope("cross_ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

writer = tf.summary.FileWriter("Training_FileWriter/")
writer1 = tf.summary.FileWriter("Validation_FileWriter/")

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()


num_epochs = 100
batch_size = 100

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
    
    # Loop over number of epochs
    for epoch in range(num_epochs):
        
        start_time = time.time()
        train_accuracy = 0
        x_train, y_true_train = load(TRAIN_DIR)
            
        # Get a batch of images and labels
        

        for batch in range(0, int(len(x_train)/batch_size)):
            
            # Get a batch of images and labels
            x_batch = x_train[batch*batch_size : batch*batch_size+batch_size] 
            y_true_batch = y_true_train[batch*batch_size : batch*batch_size+batch_size] 
            
            # print(x_batch.shape)
            # print(y_true_batch.shape)

            # Put the batch into a dict with the proper names for placeholder variables
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            
            # Run the optimizer using this batch of training data.
            sess.run(optimizer, feed_dict=feed_dict_train)
            
            # Calculate the accuracy on the batch of training data
            train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
            
            # Generate summary with the current batch of data and write to file
            summ = sess.run(merged_summary, feed_dict=feed_dict_train)
            writer.add_summary(summ, epoch*int(len(x_train)/batch_size) + batch)
        
          
        train_accuracy /= int(len(x_train)/batch_size)
        
        # Generate summary and validate the model on the entire validation set

        X_test, Y_test = load(TEST_DIR)
        test_size = int(len(X_test) * 0.5)

        X_test = X_test[:test_size]
        Y_test = Y_test[:test_size]
        
        summ, test_accuracy = sess.run([merged_summary, accuracy], feed_dict={x:X_test, y_true:Y_test})
        writer1.add_summary(summ, epoch)
        

        end_time = time.time()
        
        print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
        print("\tAccuracy:")
        #print ("\t- Training Accuracy:\t{}".format(train_accuracy))
        print ("\t- Test Accuracy:\t{}".format(test_accuracy))

'''

# example of loading the mnist dataset
from keras.datasets import mnist
from matplotlib import pyplot
from keras.utils import to_categorical
# # load dataset
# (trainX, trainy), (testX, testy) = mnist.load_data()
# # summarize loaded dataset
# print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
# print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# # plot first few images
# for i in range(9):
# 	# define subplot
# 	pyplot.subplot(330 + 1 + i)
# 	# plot raw pixel data
# 	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
# # show the figure
# pyplot.show()

# for i in range(9):
# 	pyplot.subplot(330 + 1 + i)
# 	# plot raw pixel data
# 	pyplot.imshow(train_data[i][0], cmap=pyplot.get_cmap('gray'))
# # show the figure
# pyplot.show()
# baseline cnn model for mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
def load_dataset():
	# load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()

    print(trainY)
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    print('Train: X = %s, Y = %s' % (trainX.shape, trainY.shape))
    print('Test: X = %s, Y = %s' % (testX.shape, testY.shape))
   # print(trainX[0])
   # print(trainY[0])

    return trainX, trainY, testX, testY

load_dataset()
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

define_model()