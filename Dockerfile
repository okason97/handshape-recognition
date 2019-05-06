FROM tensorflow/tensorflow:latest-py3-jupyter

ADD . /develop

COPY notebooks /tf/notebooks

EXPOSE 8000
