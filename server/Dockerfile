FROM nvcr.io/nvidia/tritonserver:22.08-py3

# Maintainer
LABEL maintainer="fname.lname@domain"

# install opencv & python-venv reqs
RUN apt-get update --no-install-recommends\
 && apt-get install libsm6 libxext6 libgl1-mesa-glx python3-venv -y

# set work directory
WORKDIR /app

# setup virtual env for python
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install python dependencies
RUN pip install --upgrade pip
COPY requirements_server.txt .
RUN pip install -r requirements_server.txt

# copy req files to app
COPY . /app/
CMD ["/bin/bash", "scripts/start_servers.sh"]
