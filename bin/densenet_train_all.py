import src.dense_net.train as train

config = {
    'data.dataset_name': ['lsa16','rwth'], 
    'data.rotation_range': [0,5,25], 
    'data.width_shift_range': [0,0.2], 
    'data.height_shift_range': [0,0.2], 
    'data.horizontal_flip': [True,False], 
    'model.growth_rate': [32,64,128], 
    'model.nb_layers': [[6,12],[6,12,16],[6,12,24,16]],
    'model.reduction': [0,0.2],
}

for dataset_name in config['data.dataset_name']:
    for rotation_range in config['data.rotation_range']:
        for width_shift_range in config['data.width_shift_range']:
            for height_shift_range in config['data.height_shift_range']:
                for horizontal_flip in config['data.horizontal_flip']:
                    for growth_rate in config['model.growth_rate']:
                        for nb_layers in config['model.nb_layers']:
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
