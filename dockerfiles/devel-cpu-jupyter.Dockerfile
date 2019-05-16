FROM ulisesjeremias/tf-docker:devel-cpu-jupyter

ADD . /develop
COPY notebooks /tf/notebooks

RUN apt-get update -q
RUN apt-get install -y git

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 --version

RUN chmod -R a+rwx /tf/

RUN git clone --branch=develop https://github.com/midusi/handshape_datasets.git /tf/lib/handshape_datasets
RUN pip3 install /tf/lib/handshape_datasets

WORKDIR /tf
