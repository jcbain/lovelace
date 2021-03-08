# FROM tensorflow/tensorflow:1.5.1-devel-py3
FROM tensorflow/tensorflow:1.15.2
ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
ADD . /app
WORKDIR /app
CMD "bash"
