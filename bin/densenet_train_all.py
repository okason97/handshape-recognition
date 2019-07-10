#!/usr/bin/env python

from src.dense_net.train import train_densenet as train
# import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#   except RuntimeError as e:
     # Visible devices must be set before GPUs have been initialized
#     print(e)

config = {
    'data.dataset_name': ['ciarp', 'rwth'], 
    'data.rotation_range': [0,30], 
    'data.width_shift_range': [0,0.2], 
    'data.height_shift_range': [0,0.2], 
    'data.horizontal_flip': [False,True], 
    'model.growth_rate': [[16,32,64],[16,32,64],[32,64,128]], 
    'model.nb_layers': [[6,12,24,16],[6,12,16],[6,12]],
    'model.reduction': [0,0.5],
}

for dataset_name in config['data.dataset_name']:
    for i in range(2):
        rotation_range = config['data.rotation_range'][i]
        width_shift_range = config['data.width_shift_range'][i]
        height_shift_range = config['data.height_shift_range'][i]
        horizontal_flip = config['data.horizontal_flip'][i]
        for i in range(3):
            nb_layers = config['model.nb_layers'][i]
            for growth_rate in config['model.growth_rate'][i]:
                for reduction in config['model.reduction']:
                    try:
                        train(dataset_name=dataset_name,rotation_range=rotation_range,
                              width_shift_range=width_shift_range, height_shift_range= height_shift_range,
                              horizontal_flip=horizontal_flip,growth_rate=growth_rate,
                              nb_layers=nb_layers,reduction=reduction, batch_size=32)
                    except:
                        try:
                            train(dataset_name=dataset_name,rotation_range=rotation_range,
                                  width_shift_range=width_shift_range, height_shift_range= height_shift_range,
                                  horizontal_flip=horizontal_flip,growth_rate=growth_rate,
                                  nb_layers=nb_layers,reduction=reduction, batch_size=16)
                        except:
                            try:
                                train(dataset_name=dataset_name,rotation_range=rotation_range,
                                      width_shift_range=width_shift_range, height_shift_range= height_shift_range,
                                      horizontal_flip=horizontal_flip,growth_rate=growth_rate,
                                      nb_layers=nb_layers,reduction=reduction, batch_size=8)
                            except:
                                print("Error with {}, growth: {}, reduction: {}. Probably memory".format(nb_layers, growth_rate, reduction))
                    finally:
                        print("Finished densenet with")
                        print("dataset_name: {}".format(dataset_name))
                        print("rotation_range: {}".format(rotation_range))
                        print("width_shift_range: {}".format(width_shift_range))
                        print("height_shift_range: {}".format(height_shift_range))
                        print("horizontal_flip: {}".format(horizontal_flip))
                        print("growth_rate: {}".format(growth_rate))
                        print("nb_layers: {}".format(nb_layers))
                        print("reduction: {}".format(reduction))
