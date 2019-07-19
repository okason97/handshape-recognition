import argparse
import configparser

from eval_setup import eval

parser = argparse.ArgumentParser(description="Run evaluation")


def preprocess_config(c):
    conf_dict = {}
    int_params = ["data.test_way", "data.test_support", "data.test_query",
                  "data.query", "data.support", "data.way", "data.episodes",
                  "data.gpu", "data.cuda", "train.patience", "model.nb_layers", "model.nb_filters"]
    float_params = ["data.rotation_range", "data.width_shift_range", "data.height_shift_range", "data.train_size", "data.test_size"]
    for param in c:
        if param in int_params:
            conf_dict[param] = int(c[param])
        elif param in float_params:
            conf_dict[param] = float(c[param])
        else:
            conf_dict[param] = c[param]
    return conf_dict


parser = argparse.ArgumentParser(description="Run evaluation")
parser.add_argument("--config", type=str, default="./src/proto_net/config/config_lsa16.conf",
                    help="Path to the config file.")

parser.add_argument("--data.dataset", type=str, default=None)
parser.add_argument("--data.split", type=str, default=None)
parser.add_argument("--data.test_way", type=int, default=None)
parser.add_argument("--data.test_support", type=int, default=None)
parser.add_argument("--data.test_query", type=int, default=None)
parser.add_argument("--data.episodes", type=int, default=None)
parser.add_argument("--data.cuda", type=int, default=None)
parser.add_argument("--data.gpu", type=int, default=None)

parser.add_argument("--data.rotation_range", type=float, default=None)
parser.add_argument("--data.width_shift_range", type=float, default=None)
parser.add_argument("--data.height_shift_range", type=float, default=None)
parser.add_argument("--data.horizontal_flip", type=bool, default=None)
parser.add_argument("--data.train_size", type=float, default=None)
parser.add_argument("--data.test_size", type=float, default=None)

parser.add_argument("--model.x_dim", type=str, default=None)
parser.add_argument("--model.type", type=str, default=None)
parser.add_argument("--model.save_path", type=str, default=None)
parser.add_argument("--model.nb_layers", type=int, default=None)
parser.add_argument("--model.nb_filters", type=int, default=None)

# Run test
args = vars(parser.parse_args())
config = configparser.ConfigParser()
config.read(args["config"])
filtered_args = dict((k, v) for (k, v) in args.items() if not v is None)
config = preprocess_config({ **config["EVAL"], **filtered_args })
eval(config)
