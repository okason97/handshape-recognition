import os
import glob
import numpy as np
import tensorflow as tf
import handshape_datasets as hd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader(object):
    def __init__(self, data, n_classes, n_way, n_support, n_query):
        self.data = data
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query

    def get_next_episode(self):
        n_examples = self.data.shape[1]
        support = np.zeros([self.n_way, self.n_support, 132, 92, 3], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, 132, 92, 3], dtype=np.float32)
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            selected = np.random.permutation(n_examples)[:self.n_support + self.n_query]
            support[i] = self.data[i_class, selected[:self.n_support]]
            query[i] = self.data[i_class, selected[self.n_support:]]

        return support, query

def load_rwth(data_dir, config, splits):
    """
    Load rwth dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset

    """

    DATASET_NAME = "rwth"
    DATASET_PATH = "/develop/data/rwth/data"

    loadedData = hd.load(DATASET_NAME, DATASET_PATH)

    features = loadedData[0]
    classes = loadedData[1]['y']
    uniqueClasses, imgsPerClass = np.unique(classes, return_counts=True)

    for c in imgsPerClass:
        print(c)

    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        classes,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        stratify=classes)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print(amountPerTest, amountPerTrain)

    if config['model.type'] in ['processed']:
        train_datagen_args = {
            'featurewise_center': True,
            'featurewise_std_normalization': True,
            'rotation_range': 10,
            'width_shift_range': 0.10,
            'height_shift_range': 0.10,
            'horizontal_flip': True,
            'fill_mode': 'constant',
            'cval': 0
        }
    else:
        train_datagen_args = {
            'featurewise_center': True,
            'featurewise_std_normalization': True
        }

    train_datagen = ImageDataGenerator(train_datagen_args)
    train_datagen.fit(x_train)

    test_datagen_args = dict(featurewise_center=True,
                             featurewise_std_normalization=True)
    test_datagen = ImageDataGenerator(test_datagen_args)
    test_datagen.fit(x_train)

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
        
        for index in i:
            x[index, :, :, :] = dg.apply_transform(x[index], dg_args)

        data = np.reshape(x, (len(uniqueClasses), amountPerClass[0], 132, 92, 3))

        data_loader = DataLoader(data,
                                 n_classes=len(uniqueClasses),
                                 n_way=n_way,
                                 n_support=n_support,
                                 n_query=n_query)

        ret[split] = data_loader

    return ret
