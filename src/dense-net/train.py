#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

import os, sys, getopt
import json
from datetime import datetime

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


from datasets import load
from densenet import densenet_model

print(tf.__version__)

def train_densenet(dataset_name = "rwth", rotation_range = 10, width_shift_range = 0.10,
          height_shift_range = 0.10, horizontal_flip = True, growth_rate = 128,
          nb_layers = [6,12], reduction = 0.0, lr = 0.001, epochs = 400,
          max_patience = 25, batch_size= 16, checkpoints = False):

    # log
    log_freq = 1
    save_freq = 40
    models_directory = 'models/'
    results_directory = 'results/'
    config_directory = 'config/'

    general_directory = "/develop/results/"
    save_directory = general_directory + "{}/dense-net/".format(dataset_name)
    results = 'epoch,loss,accuracy,test_loss,test_accuracy\n'

    date = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
    identifier = "{}-growth-{}-densenet-{}".format(
        '-'.join([str(i) for i in nb_layers]),
        growth_rate, 
        dataset_name) + date

    csv_output_map_file = save_directory + dataset_name + "_densenet.csv"
    summary_file = save_directory + 'summary.csv'

    print("hyperparameters set")
    #print(tf.test.is_gpu_available())

    x, y = load(dataset_name)

    image_shape = np.shape(x)[1:]

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.33,
                                                        random_state=42)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    n_classes = len(np.unique(y))

    '''
    class_weights = compute_class_weight('balanced', 
                                        np.unique(y),
                                        y)
    '''
    
    print("data loaded")

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        fill_mode='constant',
        cval=0)
    datagen.fit(x_train)

    test_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        fill_mode='constant',
        cval=0)
    test_datagen.fit(x_train)

    model = densenet_model(classes=n_classes, shape=image_shape, growth_rate=growth_rate, nb_layers=nb_layers, reduction=reduction)

    print("model created")

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(tf.cast(images, tf.float32), training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(tf.cast(images, tf.float32), training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    # create summary writers
    train_summary_writer = tf.summary.create_file_writer(save_directory + 'summaries/train/' + identifier)
    test_summary_writer = tf.summary.create_file_writer(save_directory +  'summaries/test/' + identifier)

    # create data generators
    train_gen =  datagen.flow(x_train, y_train, batch_size=batch_size)
    test_gen = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

    print("starting training")

    min_loss = 100
    min_loss_acc = 0
    patience = 0

    for epoch in range(epochs):
        batches = 0
        for images, labels in train_gen:
            train_step(images, labels)
            batches += 1
            if batches >= len(x_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        batches = 0
        for test_images, test_labels in test_gen:
            test_step(test_images, test_labels)
            batches += 1
            if batches >= len(x_test) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        if (epoch % log_freq == 0):
            results += '{},{},{},{},{}\n'.format(epoch,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100)
            print ('Epoch: {}, Train Loss: {}, Train Acc:{}, Test Loss: {}, Test Acc: {}'.format(epoch,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100))

            if (test_loss.result() < min_loss):    
                if not os.path.exists(save_directory + models_directory):
                    os.makedirs(save_directory + models_directory)
                # serialize weights to HDF5
                model.save_weights(save_directory + models_directory + "best{}.h5".format(identifier))
                min_loss = test_loss.result()
                min_loss_acc = test_accuracy.result()
                patience = 0
            else:
                patience += 1

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
                train_loss.reset_states()           
                train_accuracy.reset_states()           

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
                test_loss.reset_states()           
                test_accuracy.reset_states()   
                
        if checkpoints and epoch % save_freq == 0:
            if not os.path.exists(save_directory + models_directory):
                os.makedirs(save_directory + models_directory)
            # serialize weights to HDF5
            model.save_weights(save_directory + models_directory+"{}_epoch{}.h5".format(identifier,epoch))
            
        if patience >= max_patience:
            break

    if not os.path.exists(save_directory + results_directory):
        os.makedirs(save_directory + results_directory)
    file = open(save_directory + results_directory + 'results-'+ identifier + '.csv','w') 
    file.write(results) 
    file.close()

    if not os.path.exists(save_directory + config_directory):
        os.makedirs(save_directory + config_directory)

    config = {
        'data.dataset_name': dataset_name, 
        'data.rotation_range': rotation_range, 
        'data.width_shift_range': width_shift_range, 
        'data.height_shift_range': height_shift_range, 
        'data.horizontal_flip': horizontal_flip, 
        'model.growth_rate': growth_rate, 
        'model.nb_layers': nb_layers, 
        'model.reduction': reduction, 
        'train.lr': lr, 
        'train.epochs': epochs, 
        'train.max_patience': max_patience, 
        'train.batch_size': batch_size, 
    }

    with open(save_directory + config_directory + '.json', 'w') as json_file:
        json.dump(config, json_file)

    file = open(summary_file, 'a+') 
    summary = "{}, {}, dense-net, {}, {}, {}".format(
        date, dataset_name, save_directory + config_directory, min_loss, min_loss_acc)
    file.write(summary)

    file.close()