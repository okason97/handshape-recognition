import os
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
    
    return data[0], data[1]['y']