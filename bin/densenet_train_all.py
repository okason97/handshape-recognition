#!/usr/bin/env python

from src.dense_net.train import train_densenet as train

config = {
    'data.dataset_name': ['ciarp', 'lsa16', 'rwth'], 
    'data.test_size': 0.25,
    'data.train_size': [0.33, 0.5, 0.64, 0.75],
    'data.rotation_range': [0,30],
    'data.width_shift_range': [0,0.2], 
    'data.height_shift_range': [0,0.2], 
    'data.horizontal_flip': [False,True], 
    'model.growth_rate': [[128,64],[64,64]], 
    'model.nb_layers': [[[6,12],[6,12]],[[6,12],[6,12]]],
    'model.reduction': 0.5,
}

for i in range(3):
    for j in range(2):
        rotation_range = config['data.rotation_range'][j]
        width_shift_range = config['data.width_shift_range'][j]
        height_shift_range = config['data.height_shift_range'][j]
        horizontal_flip = config['data.horizontal_flip'][j]
        dataset_name = config['data.dataset_name'][i]
        growth_rate = config['model.growth_rate'][i][j]
        nb_layers = config['model.nb_layers'][i][j]
        reduction = config['model.reduction']
        test_size = config['data.test_size']
        for train_size in config['data.train_size']:                 
            train(dataset_name=dataset_name,rotation_range=rotation_range,
                    width_shift_range=width_shift_range, height_shift_range= height_shift_range,
                    horizontal_flip=horizontal_flip,growth_rate=growth_rate,
                    nb_layers=nb_layers,reduction=reduction, test_size=test_size,
                    train_size=train_size, batch_size=16)
            print("Finished densenet with")
            print("dataset_name: {}".format(dataset_name))
            print("rotation_range: {}".format(rotation_range))
            print("width_shift_range: {}".format(width_shift_range))
            print("height_shift_range: {}".format(height_shift_range))
            print("horizontal_flip: {}".format(horizontal_flip))
            print("growth_rate: {}".format(growth_rate))
            print("nb_layers: {}".format(nb_layers))
            print("reduction: {}".format(reduction)) 
            print("train size: {}".format(train_size)) 
