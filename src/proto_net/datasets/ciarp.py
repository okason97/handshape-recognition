import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import handshape_datasets as hd
from src.utils.model_selection import train_test_split_balanced

class DataLoader(object):
    def __init__(self, data, n_classes, n_way, n_support, n_query, x_dim):
        self.data = data
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query
        self.x_dim = x_dim

    def get_next_episode(self):
        n_examples = self.data.shape[1]
        w, h, c = self.x_dim
        support = np.zeros([self.n_way, self.n_support, w, h, c], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, w, h, c], dtype=np.float32)
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            selected = np.random.permutation(n_examples)[:self.n_support + self.n_query]
            support[i] = self.data[i_class, selected[:self.n_support]]
            query[i] = self.data[i_class, selected[self.n_support:]]

        return support, query

def load_ciarp(data_dir, config, splits):
    """
    Load ciarp dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset

    """

    DATASET_NAME = "ciarp"
    DATASET_PATH = "/develop/data/ciarp/data"

    data = hd.load(DATASET_NAME, DATASET_PATH)

    x_train, y_train = data['train_Kinect_WithoutGabor']
    x_test, y_test = data['test_Kinect_WithoutGabor']

    X = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    _, uniqueClasses = np.unique(y, return_counts=True)

    _, amountPerTrain = np.unique(y_train, return_counts=True)
    _, amountPerTest = np.unique(y_test, return_counts=True)

    x_train, x_test, y_train, y_test = train_test_split_balanced(X,
                                                                 y,
                                                                 test_size=0.33,
                                                                 n_train_per_class=np.min(amountPerTrain),
                                                                 n_test_per_class=np.min(amountPerTest))
    x_train, x_test = x_train / 255.0, x_test / 255.0

    _, amountPerTrain = np.unique(y_train, return_counts=True)
    _, amountPerTest = np.unique(y_test, return_counts=True)

    train_datagen_args = dict(featurewise_center=True,
                              featurewise_std_normalization=True,
                              rotation_range=config['data.rotation_range'],
                              width_shift_range=config['data.width_shift_range'],
                              height_shift_range=config['data.height_shift_range'],
                              horizontal_flip=config['data.horizontal_flip'],
                              fill_mode='constant',
                              cval=0)
    train_datagen = ImageDataGenerator(train_datagen_args)
    train_datagen.fit(x_train)

    test_datagen_args = dict(featurewise_center=True,
                             featurewise_std_normalization=True,
                             fill_mode='constant',
                             cval=0)
    test_datagen = ImageDataGenerator(test_datagen_args)
    test_datagen.fit(x_train)

    w, h, c = list(map(int, config['model.x_dim'].split(',')))

    ret = {}
    for split in splits:
        # n_way (number of classes per episode)
        if split in ['val', 'test']:
            n_way = config['data.test_way']
        else:
            n_way = config['data.train_way']

        # n_support (number of support examples per class)
        if split in ['val', 'test']:
            n_support = config['data.test_support']
        else:
            n_support = config['data.train_support']

        # n_query (number of query examples per class)
        if split in ['val', 'test']:
            n_query = config['data.test_query']
        else:
            n_query = config['data.train_query']

        if split in ['val', 'test']:
            y = y_test
            x = x_test
            dg = train_datagen
            dg_args = train_datagen_args
        else:
            y = y_train
            x = x_train
            dg = test_datagen
            dg_args = test_datagen_args

        amountPerClass = amountPerTest if split in ['val', 'test'] else amountPerTrain

        i = np.argsort(y)
        x = x[i, :, :, :]
        
        if config['model.type'] in ['processed']:
            for index in i:
                x[index, :, :, :] = dg.apply_transform(x[index], dg_args)

        data = np.reshape(x, (len(uniqueClasses), amountPerClass[0], w, h, c))

        data_loader = DataLoader(data,
                                 n_classes=len(uniqueClasses),
                                 n_way=n_way,
                                 n_support=n_support,
                                 n_query=n_query,
                                 x_dim=(w, h, c))

        ret[split] = data_loader

    return ret
