import numpy as np

def train_test_split_balanced(data, target, test_size=0.2, train_size=0, n_train_per_class=0, n_test_per_class=0):
    classes = np.unique(target)

    # can give test_size as fraction of input data size of number of samples
    if test_size < 1:
        n_test = np.round(len(target)*test_size)
    else:
        n_test = test_size
    
    if train_size < 1:
        n_train = np.round(len(target)*train_size)
    else:
        n_train = train_size

    # variables for manual balance
    n_train_per_class = int(n_train_per_class)
    n_test_per_class = int(n_test_per_class)

    if n_test_per_class <= 0:
        n_train_per_class = max(1, int(np.floor(n_train / len(classes))))

    if n_test_per_class <= 0:
        n_test_per_class = max(1, int(np.floor(n_test / len(classes))))

    ixs = []
    for cl in classes:
        if (n_train_per_class+n_test_per_class) > np.sum(target == cl):
            # if data has too few samples for this class, do upsampling
            # split the data to training and testing before sampling so data points won't be
            # shared among training and test data
            splitix = int(np.ceil(n_train_per_class/(n_train_per_class+n_test_per_class)*np.sum(target == cl)))
            ixs.append(np.r_[np.random.choice(np.nonzero(target == cl)[0][:splitix], n_train_per_class),
                       np.random.choice(np.nonzero(target == cl)[0][splitix:], n_test_per_class)])
        else:
            ixs.append(np.random.choice(np.nonzero(target == cl)[0],
                                        n_train_per_class+n_test_per_class,
                                        replace=False))

    # take same num of samples from all classes
    ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])

    X_train = data[ix_train, :]
    X_test = data[ix_test, :]
    y_train = target[ix_train]
    y_test = target[ix_test]

    return X_train, X_test, y_train, y_test
