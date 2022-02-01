#!/bin/bash
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

echo "$gpunumber"
echo "$port"

echo "Stopping and removing docker container 'model_server_container' if it is running"
docker stop model_server_container || true
docker rm model_server_container || true

echo "Docker Container starting with FastAPI port: $port"
docker run \
      --rm \
      --gpus device=$gpunumber \
      -p 0.0.0.0:$port:8080 \
      --name model_server_container \
      --env LANG=en_US.UTF-8 \
      tf_model_server
