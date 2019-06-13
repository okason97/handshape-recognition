#!/usr/bin/env python
# coding: utf-8

# In[13]:


import tensorflow as tf

import numpy as np
from tensorflow.keras.layers import Input, ZeroPadding2D, Dense, Dropout, Activation, Convolution2D, Reshape
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization

from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os, sys, getopt

import json

from datetime import datetime

import handshape_datasets as hd

from sklearn.model_selection import train_test_split

from densenet import densenet_model

print(tf.__version__)


# In[14]:

argv = sys.argv

# hyperparameters
# data
dataset_name = "rwth"
rotation_range=10
width_shift_range=0.10
height_shift_range=0.10
horizontal_flip=True

# model
growth_rate=128
nb_layers=[6,12]

# training
lr = 0.001
epochs = 400
max_patience = 25

# log
log_freq = 1
save_freq = 40
models_directory = 'models/'
results_directory = 'results/'
config_directory = 'config/'

try:
    opts, args = getopt.getopt(argv,"h",["dataset=","rotation=","w-shift=","h-shift=","h-flip=","growth-r=","nb-layers=","lr=","epochs=","patience=","log-freq=","save-freq=","models-dir=","results_dir="])
except getopt.GetoptError:
    print('test.py --dataset=rwth --rotation=10 --w-shift=0.1 --h-shift=0.1 h-flip=True growth-r=128 nb-layers=6:12 lr=0.001 epochs=400 patience=25 log-freq=1 save-freq=40 models-dir=models results_dir=results')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('test.py --dataset=rwth --rotation=10 --w-shift=0.1 --h-shift=0.1 h-flip=True growth-r=128 nb-layers=6:12 lr=0.001 epochs=400 patience=25 log-freq=1 save-freq=40 models-dir=models results_dir=results')
        sys.exit()
    elif opt == "--dataset":
        dataset_name = arg
    elif opt == "--rotation":
        rotation_range = int(arg)
    elif opt == "--w-shift":
        width_shift_range = float(arg)
    elif opt == "--h-shift":
        height_shift_range = float(arg)
    elif opt == "--h-flip":
        if arg in ("True", "true"):
            horizontal_flip = True
        elif arg in ("False", "false"):
            horizontal_flip = False
    elif opt == "--growth-r":
        growth_rate = int(arg)
    elif opt == "--nb-layers":
        nb_layers = [int(n) for n in arg.split(":")]
    elif opt == "--lr":
        lr = float(arg)
    elif opt == "--epochs":
        epochs = int(arg)
    elif opt == "--patience":
        max_patience = int(arg)
    elif opt == "--log-freq":
        log_freq = int(arg)
    elif opt == "--save-freq":
        save_freq = int(arg)
    elif opt == "--models-dir":
        models_directory = arg
    elif opt == "--results_dir":
        results_directory = arg

general_directory = "/results/"
save_directory = general_directory + "{}/densenet/".format(dataset_name)
results = 'epoch,loss,accuracy,test_loss,test_accuracy\n'
date = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
identifier = "{}-growth-{}-densenet-{}".format(
    '-'.join([str(i) for i in nb_layers] ), 
    growth_rate, 
    dataset_name) + date
csv_output_map_file = save_directory + dataset_name + "_densenet.csv"
summary_file = save_directory + 'summary.csv'
print("hyperparameters set")

# In[2]:


print(tf.test.is_gpu_available())


# In[3]:


data = hd.load(dataset_name)


# In[4]:


features = data[0]
labels = data[1]['y']
n_classes = len(np.unique(labels))
image_shape = np.shape(data[0])[1:]

x_train, x_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    test_size=0.33,
                                                    random_state=42)
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[5]:


print("data loaded")


# In[6]:


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


# In[7]:


model = densenet_model(classes=n_classes, shape=image_shape, growth_rate=growth_rate, nb_layers=nb_layers)

print("model created")


# In[8]:


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()


# In[9]:


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# In[10]:


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(tf.cast(images, tf.float32), training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


# In[11]:


@tf.function
def test_step(images, labels):
    predictions = model(tf.cast(images, tf.float32), training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


# In[12]:


# create summary writers
train_summary_writer = tf.summary.create_file_writer(save_directory + 'summaries/train/' + identifier)
test_summary_writer = tf.summary.create_file_writer(save_directory +  'summaries/test/' + identifier)

# create data generators
train_gen =  datagen.flow(x_train, y_train, batch_size=16)
test_gen = test_datagen.flow(x_test, y_test, batch_size=16, shuffle=False)

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
            
    if (epoch % save_freq == 0):
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
    'train.lr': lr, 
    'train.epochs': epochs, 
    'train.max_patience': max_patience, 
}
with open(save_directory + config_directory + '.json', 'w') as json_file:
  json.dump(config, json_file)

file = open(summary_file,'a+') 
summary = "{}, {}, densenet, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
    date, dataset_name, save_directory + config_directory, min_loss, min_loss_acc)
file.write(summary) 
file.close()
