FROM tensorflow/tensorflow:2.12.0-gpu

# maintainer
LABEL maintainer="fname.lname@domain.com"

# set username & uid inside docker
ARG UNAME=user1
ARG UID=1000
ENV WORKDIR="/home/$UNAME/tensorflow_training"

# add user UNAME as a member of the sudoers group
RUN useradd -rm --home-dir "/home/$UNAME" --shell /bin/bash -g root -G sudo -u "$UID" "$UNAME"

# set workdir
WORKDIR "$WORKDIR"

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# setup virtual env for python
ENV VIRTUAL_ENV="/home/$UNAME/venv"
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# freq changing files are added below
COPY . "$WORKDIR"

# change file ownership to docker user
RUN chown -R "$UNAME" "$WORKDIR"

USER "$UNAME"
