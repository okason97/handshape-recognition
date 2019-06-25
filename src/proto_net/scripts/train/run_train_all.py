import configparser

from train_setup import train

configs = {
    'lsa16': {
        'data.train_way': [5, 16],
        'data.test_way': [5],
        # done (1, 1, 1, 1) in previous experiments
        'data.support_query': [(1, 15, 1, 1), (5, 5, 5, 5), (5, 5, 5, 11), (5, 25, 5, 5)],

        #'data.rotation_range': [0, 25], 
        #'data.width_shift_range': [0.1], 
        #'data.height_shift_range': [0.1],
        #'data.horizontal_flip': [True, False],
        'data.args': [(0, 0, 0, False), (24, 0.2, 0.2, True)],

        'model.type': ['expr'],
        'model.nb_layers': [4, 8],
        'model.nb_filters': [128, 64],

        'train.lr': [0.001]
    },
    'rwth': {
        'data.train_way': [20, 5],
        'data.test_way': [5],
        'data.support_query': [(1, 1, 1, 1), (5, 5, 5, 5), (1, 19, 1, 1), (5, 15, 5, 5)],
        
        #'data.rotation_range': [0, 25], 
        #'data.width_shift_range': [0.1], 
        #'data.height_shift_range': [0.1],
        #'data.horizontal_flip': [True, False],
        'data.args': [(0, 0, 0, False), (24, 0.2, 0.2, True)],

        'model.type': ['expr'],
        'model.nb_layers': [4, 8],
        'model.nb_filters': [128, 64],

        'train.lr': [0.001]
    }
}


def preprocess_config(c):
    conf_dict = {}
    int_params = ['data.train_way', 'data.test_way', 'data.train_support',
                  'data.test_support', 'data.train_query', 'data.test_query',
                  'data.episodes', 'data.gpu', 'data.cuda', 'model.z_dim', 
                  'train.epochs', 'train.patience']
    float_params = ['train.lr', 'data.rotation_range',
                    'data.width_shift_range', 'data.height_shift_range']
    for param in c:
        if param in int_params:
            conf_dict[param] = int(c[param])
        elif param in float_params:
            conf_dict[param] = float(c[param])
        else:
            conf_dict[param] = c[param]
    return conf_dict


for dataset in ['rwth']:
    config_from_file = configparser.ConfigParser()
    config_from_file.read("./src/proto_net/config/config_{}.conf".format(dataset))

    ds_config = configs[dataset]

    for train_way in ds_config['data.train_way']:
        for test_way in ds_config['data.test_way']:
            for train_support, train_query, test_support, test_query in ds_config['data.support_query']:
                for rotation_range, width_shift_range, height_shift_range, horizontal_flip in ds_config['data.args']:
                    for model_type in ds_config['model.type']:
                        for nb_layers in ds_config['model.nb_layers']:
                            for nb_filters in ds_config['model.nb_filters']:
                                for lr in ds_config['train.lr']:
                                    custom_params = {
                                        'data.train_way': train_way,
                                        'data.train_support': train_support,
                                        'data.train_query': train_query,
                                        'data.test_way': test_way,
                                        'data.test_support': test_support,
                                        'data.test_query': test_query,

                                        'data.rotation_range': rotation_range, 
                                        'data.width_shift_range': width_shift_range,
                                        'data.height_shift_range': height_shift_range,
                                        'data.horizontal_flip': horizontal_flip, 

                                        'model.type': model_type,
                                        'model.nb_layers': nb_layers,
                                        'model.nb_filters': nb_filters,

                                        'train.lr': lr
                                    }

                                    preprocessed_config = preprocess_config({ **config_from_file['TRAIN'], **custom_params })
                                    train(preprocessed_config)
