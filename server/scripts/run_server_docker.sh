#!/bin/bash

def_cont_name=model_server_container
helpFunction()
{
   echo ""
   echo "Usage: $0 -g gpunumber -p port"
   echo -e "\t-g gpu number"
   echo -e "\t-p port"
   exit 1 # Exit script after printing help
}

while getopts "g:p:" opt
do
   case "$opt" in
      g ) gpunumber="$OPTARG" ;;
      p ) port="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$gpunumber" ] || [ -z "$port" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Check if the container is running
if [ "$(docker ps -q -f name=$def_cont_name)" ]; then
    echo "Stopping docker container '$def_cont_name'"
    docker stop "$def_cont_name"
    echo "Stopped container '$def_cont_name'"
fi

echo "Docker Container starting with port exposed at port: $port on gpu: $gpu"
docker run \
      --rm -d \
      --gpus device="$gpunumber" \
      -p 127.0.0.1:"$port":8080 \
      --name $def_cont_name \
      --env LANG=en_US.UTF-8 \
      tf_model_server
