#!/bin/bash

echo "Building Docker Container for model inference"
docker build -t tf_model_server .
