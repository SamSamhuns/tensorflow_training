#!/bin/bash
docker build -t tensorflow_train:latest --build-arg UID=$(id -u) .
