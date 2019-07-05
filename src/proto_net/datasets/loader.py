from .ciarp import load_ciarp
from .lsa16 import load_lsa16
from .rwth import load_rwth

def load(data_dir, config, splits):
    """
    Load specific dataset.

    Args:
        data_dir (str): path to the dataset directory.
        config (dict): general dict with settings.
        splits (list): list of strings 'train'|'val'|'test'.

    Returns (dict): dictionary with keys 'train'|'val'|'test'| and values
    as tensorflow Dataset objects.

    """

    if config['data.dataset'] == "ciarp":
        ds = load_ciarp(data_dir, config, splits)
    elif config['data.dataset'] == "lsa16":
        ds = load_lsa16(data_dir, config, splits)
    elif config['data.dataset'] == "rwth":
        ds = load_rwth(data_dir, config, splits)
    else:
        raise ValueError(f"Unknow dataset: {config['data.dataset']}")

    return ds
