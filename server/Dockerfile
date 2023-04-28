FROM nvcr.io/nvidia/tritonserver:22.08-py3

# Maintainer
LABEL maintainer="fname.lname@domain"

# install opencv & python-venv reqs
RUN apt-get update --no-install-recommends \
 && apt-get install libsm6 libxext6 libgl1-mesa-glx python3-venv -y

# remove cache
RUN apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# set username & uid inside docker
ARG UNAME=triton-server
ENV WORKDIR="/home/$UNAME/app"

# set workdir
WORKDIR ${WORKDIR}

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# setup virtual env for python
ENV VIRTUAL_ENV="/home/$UNAME/venv"
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python dependencies
RUN pip install pip==23.1.2
COPY ./requirements_server.txt "$WORKDIR/app/requirements_server.txt"
RUN pip install --no-cache-dir --default-timeout=100 -r "$WORKDIR/app/requirements_server.txt"

# freq changing files are copied below to app
COPY . "$WORKDIR"

# change file ownership to docker user
RUN chown -R "$UNAME" "$WORKDIR"

USER "$UNAME"
CMD ["/bin/bash", "scripts/start_servers.sh"]
