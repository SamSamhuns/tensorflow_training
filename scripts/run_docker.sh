#!/bin/bash

def_cont_name=tf_train_container

helpFunction()
{
   echo ""
   echo "Usage: $0 -p port"
   echo -e "\t-p tensorboard_port"
   exit 1 # Exit script after printing help
}

while getopts "p:" opt
do
   case "$opt" in
      p ) port="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$port" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Check if the container is running
if [ "$(docker ps -q -f name=$def_cont_name)" ]; then
    # Stop the container
    echo "Stopping docker container '$def_cont_name'"
    docker stop "$def_cont_name"
    echo "Stopped container '$def_cont_name'"
fi

docker run \
      -ti --rm \
      -p 0.0.0.0:"$port":6006 \
      -v "$PWD"/checkpoints:/tensorflow_training/checkpoints \
      -v "$PWD"/data:/tensorflow_training/data \
      --name "$def_cont_name" \
      tensorflow_train \
      bash
