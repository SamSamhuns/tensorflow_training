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

echo "Building tf model server image and starting container"

# cd to server dir and create docker file and run dockerfile to start server
cd server
bash scripts/build_server_docker.sh
bash scripts/run_server_docker.sh -g "$gpunumber" -p "$port"
