FROM tensorflow/tensorflow:1.5.1-devel-py3
ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
ADD . /app
WORKDIR /app
CMD "/bin/sh"
