FROM ubuntu:16.04

MAINTAINER Rejith.Raghavan@disney.com



RUN apt-get update && \
    apt-get install -y curl build-essential libffi-dev python-pip && \
    apt-get clean

RUN apt-get update && \
    apt-get install -y  ipython ipython-notebook && \
    apt-get clean


RUN pip install --ignore-installed --upgrade  https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0-cp27-none-linux_x86_64.whl

RUN pip install tflearn
RUN pip install tqdm
RUN pip install opencv-python

# Keras
RUN pip install keras
RUN pip install tflearn

RUN pip install IPython==5.0
RUN pip install jupyter


RUN apt-get update && \
    apt-get install -y   libglib2.0-0 && \
    apt-get clean


RUN pip install matplotlib


EXPOSE 8888 6006

RUN mkdir /notebooks
VOLUME ["/notebooks"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--notebook-dir=/notebooks"]

