FROM tensorflow/tensorflow:latest-py3-jupyter

ADD . /develop

COPY notebooks /tf/notebooks

RUN pip install --upgrade pip && \
    pip3 uninstall -y tensorflow && \
    pip3 install tensorflow==2.0.0-alpha0
    
RUN apt-get update -q
RUN apt-get install -y software-properties-common python-software-properties
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -q
RUN apt-get install -y python3.6 git

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 --version

RUN git clone --branch=develop https://github.com/midusi/handshape_datasets.git lib/handshape_datasets
RUN cd lib/handshape_datasets

EXPOSE 8000
