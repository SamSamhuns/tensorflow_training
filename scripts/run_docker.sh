#!/bin/bash
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

echo "Starting tf model training container with tensorboard exposed on port $port"
docker run \
      -ti --rm \
      -p 0.0.0.0:$port:6006 \
      -v $PWD/checkpoints:/tensorflow_training/checkpoints \
      -v $PWD/data:/tensorflow_training/data \
      --name tf_train_container \
      tensorflow_train \
      bash
