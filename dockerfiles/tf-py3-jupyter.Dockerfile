ARG DOCKER_ENV=cpu

FROM ulisesjeremias/tf-docker:${DOCKER_ENV}-jupyter

ADD . /develop
COPY src /tf/notebooks

# Needed for string testing
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q
RUN apt-get install -y git nano

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 --version

RUN git clone --branch=develop https://github.com/midusi/handshape_datasets.git /tf/lib/handshape_datasets
RUN pip3 install -e /tf/lib/handshape_datasets

RUN git clone https://github.com/ulises-jeremias/prototypical-networks-tf.git /tf/lib/prototypical-networks-tf
RUN pip3 install -e /tf/lib/prototypical-networks-tf

RUN pip3 install -e /develop/src/proto-net
RUN pip3 install sklearn
RUN if [[ DOCKER_ENV == "gpu" ]]; then pip3 install tf-nightly-gpu-2.0-preview; fi

# Default dir for handshape datasets lib - use /data instead
RUN mkdir -p /.handshape_datasets
RUN chmod -R a+rwx /.handshape_datasets

WORKDIR /tf
