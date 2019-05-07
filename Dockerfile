FROM tensorflow/tensorflow:latest-py3-jupyter

ADD . /develop

COPY notebooks /tf/notebooks

RUN pip3 uninstall -y tensorflow && \
    pip3 install tensorflow==2.0.0-alpha0

EXPOSE 8000
