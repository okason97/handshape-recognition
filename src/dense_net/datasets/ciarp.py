import os
import numpy as np
import handshape_datasets as hd

def load_lsa16(dataset_name):
    """
    Load lsa16 dataset.

    Returns (x, y): as dataset x and y.

    """
    DATASET_PATH = '/develop/data/{}/data'.format(dataset_name)

    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    data = hd.load(dataset_name, DATASET_PATH)

    # TODO: define best way to do this

    x_train, y_train = data['train_Kinect_WithoutGabor']
    x_test, y_test = data['test_Kinect_WithoutGabor']

    X = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    return X, y