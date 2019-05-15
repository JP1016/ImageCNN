#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:24:52 2019

@author: jp
"""


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from random import randint
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
IMAGE_SIZE_X=28
IMAGE_SIZE_Y=28
NO_OF_LAYERS = 128 
OUTPUT_LAYERS = 10

model = keras.Sequential([
        keras.layers.Flatten(input_shape=(IMAGE_SIZE_X, IMAGE_SIZE_Y)),
        keras.layers.Dense(NO_OF_LAYERS, activation=tf.nn.relu),
        keras.layers.Dense(NO_OF_LAYERS, activation=tf.nn.relu),
        keras.layers.Dense(OUTPUT_LAYERS, activation=tf.nn.softmax)
    ])

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  
  plt.imshow(img, cmap=plt.cm.binary) #2nd parameter for grayscale image
  
  predicted_label = np.argmax(predictions_array) # get the index of the highest value by flattening the array
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


def test_image(test_images,test_labels,class_names):
    print("*"*20)
    print("Begin Prediction")
    print("*"*20)
    random_predict_index=randint(0,len(test_images));
    image=test_images[random_predict_index]
    
    #tf.keras models are optimized to make predictions on a batch, or collection,
    #of examples at once. So even though we're using a single image, we need to add it to a list:
    img = (np.expand_dims(image,0))
    
    #predict() is for the actual prediction. It generates output predictions for the input samples.
    predictions_single = model.predict(img)
    plot_value_array(0, predictions_single, test_labels)
    plt.xticks(range(10), class_names, rotation=45)
    plt.show()    
    prediction_result = np.argmax(predictions_single[0])
    plot_image(random_predict_index, predictions, test_labels, test_images)
    print("Prediction of Clothing is :",class_names[prediction_result])

def main():
    
    ################### DATASET DECLARATION AND LOADING ###################
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    #changing the values to 0-1 range
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    #using categorical classification since there are more than one output
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    ################### TRAINING THE MODEL ###################
    #fit() is for training the model with the given inputs (and corresponding training labels).
    model.fit(train_images, train_labels, epochs=10)
    
    #evaluate() is for evaluating the already trained model using the validation 
    # (or test) data and the corresponding labels. Returns the loss value and metrics values for the model.
    ################### TESTING THE MODEL ###################
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    
    print('Test accuracy:', test_acc)
    
    #Running a random test
    test_image(test_images,test_labels,class_names)

main()

