FROM tensorflow/tensorflow:2.6.1-gpu

# maintainer
LABEL maintainer="fname.lname@domain.com"

# set work directory
WORKDIR /tensorflow_training

# install python dependencies
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install --default-timeout=100 -r requirements.txt

# freq changing files are added below
COPY . /tensorflow_training
