import os
import handshape_datasets as hd
import numpy as np

def load_rwth(dataset_name):
    """
    Load rwth dataset.

    Returns (x, y): as dataset x and y.

    """
    DATASET_PATH = '/develop/data/{}/data'.format(dataset_name)

    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    data = hd.load(dataset_name, DATASET_PATH)
    
    good_min = 20
    good_classes = []
    n_unique = len(np.unique(data[1]['y']))
    for i in range(n_unique):
        images = data[0][np.equal(i,data[1]['y'])]
        if len(images) >= good_min:
            good_classes = good_classes + [i]
            
    x = data[0][np.in1d(data[1]['y'], good_classes)]
    y = data[1]['y'][np.in1d(data[1]['y'], good_classes)]
    my_dict = dict(zip(np.unique(y), range(len(np.unique(y)))))
    y = np.vectorize(my_dict.get)(y)
    
    return x, y