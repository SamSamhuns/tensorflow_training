FROM tensorflow/tensorflow:2.10.0-gpu

# maintainer
LABEL maintainer="fname.lname@domain.com"

# set work directory
WORKDIR /tensorflow_training

# setup virtual env for python
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python dependencies
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install --default-timeout=100 -r requirements.txt

# freq changing files are added below
COPY . /tensorflow_training
