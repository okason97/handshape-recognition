#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

import os, sys, getopt
import json
from datetime import datetime

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .datasets import load
from densenet import densenet_model

def eval_densenet(dataset_name = "rwth", growth_rate = 128, nb_layers = [6,12],
                  reduction = 0.0, lr = 0.001, epochs = 400,
                  max_patience = 25, batch_size = 16, checkpoints = False,
                  weight_classes = False, model_path = ""):
    
    np.random.seed(2019)
    tf.random.set_seed(2019)

    # log
    log_freq = 1

    print("hyperparameters set")

    x, y = load(dataset_name)

    image_shape = np.shape(x)[1:]

    x_train, x_test, _, y_test = train_test_split(x,
                                                  y,
                                                  test_size=0.33,
                                                  random_state=42)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    n_classes = len(np.unique(y))

    if weight_classes:
        class_weights = compute_class_weight('balanced',
                                             np.unique(y),
                                             y)
    print("data loaded")

    test_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        fill_mode='constant',
        cval=0)
    test_datagen.fit(x_train)

    model = densenet_model(classes=n_classes, shape=image_shape, growth_rate=growth_rate, nb_layers=nb_layers, reduction=reduction)
    model.load_weights(model_path)

    print("model created")

    if weight_classes:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

        def weightedLoss(originalLossFunc, weightsList):

            @tf.function
            def lossFunc(true, pred):

                axis = -1 #if channels last
                # axis=  1 #if channels first

                # argmax returns the index of the element with the greatest value
                # done in the class axis, it returns the class index    
                classSelectors = tf.argmax(true, axis=axis, output_type=tf.int32) 

                # considering weights are ordered by class, for each class
                # true(1) if the class index is equal to the weight index   
                classSelectors = [tf.equal(i, classSelectors) for i in range(len(weightsList))]

                # casting boolean to float for calculations  
                # each tensor in the list contains 1 where ground true class is equal to its index 
                # if you sum all these, you will get a tensor full of ones. 
                classSelectors = [tf.cast(x, tf.float32) for x in classSelectors]

                # for each of the selections above, multiply their respective weight
                weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

                # sums all the selections
                # result is a tensor with the respective weight for each element in predictions
                weightMultiplier = weights[0]
                for i in range(1, len(weights)):
                    weightMultiplier = weightMultiplier + weights[i]


                # make sure your originalLossFunc only collapses the class axis
                # you need the other axes intact to multiply the weights tensor
                loss = originalLossFunc(true,pred) 
                loss = loss * weightMultiplier

                return loss
            return lossFunc
        loss_object = weightedLoss(loss_object, class_weights)
    else:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def test_step(images, labels):
        predictions = model(tf.cast(images, tf.float32), training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    test_gen = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

    print("starting evaluation")

    for epoch in range(epochs):
        batches = 0
        for test_images, test_labels in test_gen:
            test_step(test_images, test_labels)
            batches += 1
            if batches >= len(x_test) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        if (epoch % log_freq == 0):
            print ('Epoch: {} Test Loss: {} Test Acc: {}'.format(epoch,
                                                                 test_loss.result(),
                                                                 test_accuracy.result()*100))

