# Sign Language Recognition

## Content

- [Models](#models)
  - [Prototypical Networks for Few-shot Learning](#prototypical-networks-for-few-shot-learning)
    - [Evaluating](#evaluating)
    - [Results](#results)
  - Dense Net
- [Quickstart](#quickstart)
- [Setup and use docker](#setup-and-use-docker)

## Models

### Prototypical Networks for Few-shot Learning

Tensorflow v2 implementation of NIPS 2017 Paper _Prototypical Networks for Few-shot Learning_.

Implementation based on [protonet](https://github.com/ulises-jeremias/prototypical-networks-tf).

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/protonet --mode train --config <config>
```

`<config> = ciarp | lsa16 | rwth`

#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/protonet --mode eval --config <config>
```

`<config> = ciarp | lsa16 | rwth, ciarp & rwth not working yet`

#### Results

In the `protonet-tf/results/<ds>` directory you can find the following results of training processes on a specific dataset `<ds>`:

-  `protonet-tf/results/<ds>/models/`, there are trained models.

-  `protonet-tf/results/<ds>/results/`, there are debug output on different `.csv` files.

-  `protonet-tf/results/<ds>/summaries/`, tensorboard summaries.

To run TensorBoard, use the following command 

```sh
$ tensorboard --logdir=./protonet-tf/results/<ds>/summaries/
```

### Dense Net

. . .

## Quickstart

```sh
$ ./bin/start [-t <tag-name>] [--sudo <bool>]
```

```
<tag-name> = cpu | devel-cpu | gpu
<bool> = false | true
```

## Setup and use docker

Build the docker image,

```sh
$ docker build --rm -f dockerfiles/tf-py3-jupiter.Dockerfile -t sign-language-recognition:latest .
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

To run TensorBoard, use the following command (alternatively python -m tensorboard.main)

```sh
$ tensorboard --logdir=/path/to/summaries
```
