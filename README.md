# Sign Language Recognition

## Models

# Prototypical Networks for Few-shot Learning

Tensorflow v2 implementation of NIPS 2017 Paper _Prototypical Networks for Few-shot Learning_.

First install [protonet](https://github.com/ulises-jeremias/prototypical-networks-tf).

Run the following command to run training on lsa16 with default parameters.

```sh
$ python protonet-tf/train/run_train.py --config protonet-tf/config_lsa16.conf
```

## Evaluating

To run evaluation on lsa16

```sh
$ python protonet-tf/eval/run_eval.py --config protonet-tf/config_lsa16.conf
```


## Quickstart

```sh
$ ./bin/start -t <tag-name>
```

`<tag-name> = cpu | devel-cpu | gpu`

## Setup and use docker

Build the docker image,

```sh
$ docker build --rm -f dockerfiles/cpu-jupiter.Dockerfile -t sign-language-recognition:latest .
```

and now run the image

```sh
$ docker run --rm -u $(id -u):$(id -g) -p 6006:6006 -p 8888:8888 sign-language-recognition:latest
```

Visit that link, hey look your jupyter notebooks are ready to be created.

If you want, you can attach a shell to the running container

```sh
$ docker exec -it <container-id> /bin/sh -c "[ -e /bin/bash ] && /bin/bash || /bin/sh"
```

And then you can find the entire source code in `/develop`.

```sh
$ cd /develop
```
