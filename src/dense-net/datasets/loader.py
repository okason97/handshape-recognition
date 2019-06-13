from .lsa16 import load_lsa16
from .rwth import load_rwth

def load(dataset_name):
    """
    Load specific dataset.

    Args:
        dataset_name (str): name of the dataset.

    Returns (x, y): as dataset x and y.

    """

    if dataset_name == "lsa16":
        x, y = load_lsa16(dataset_name)
    elif dataset_name == "rwth":
        x, y = load_rwth(dataset_name)
    else:
        raise ValueError("Unknow dataset: {}".format(dataset_name))

    return x, y
