ARG DOCKER_ENV=cpu

FROM ulisesjeremias/tf-docker:${DOCKER_ENV}-jupyter
# DOCKER_ENV are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)

ARG DOCKER_ENV

ADD . /develop
COPY src /tf/notebooks

# Needed for string testing
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y git nano

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 --version

# Clone and install handshape datasets
RUN git clone --branch=develop https://github.com/midusi/handshape_datasets.git /tf/lib/handshape_datasets
RUN pip3 install -e /tf/lib/handshape_datasets

# Clone and install proto-net
RUN git clone --branch=develop https://github.com/ulises-jeremias/prototypical-networks-tf.git /tf/lib/prototypical-networks-tf
RUN pip3 install -e /tf/lib/prototypical-networks-tf

# Install dense-net
RUN pip3 install densenet

# Install proto-net scripts and handshape-recognition utils
RUN pip3 install -e /develop/

RUN pip3 install sklearn opencv-python IPython
RUN if [[ "$DOCKER_ENV" = "gpu" ]]; then echo -e "\e[1;31mINSTALLING GPU SUPPORT\e[0;33m"; pip3 install -U tf-nightly-gpu-2.0-preview tb-nightly; fi

# Default dir for handshape datasets lib - use /data instead
RUN mkdir -p /.handshape_datasets
RUN chmod -R a+rwx /.handshape_datasets

WORKDIR /develop

CMD ["bash", "-c", "source /etc/bash.bashrc && /develop/bin/execute"]
