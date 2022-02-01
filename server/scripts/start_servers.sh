#!/bin/bash
#Run Triton on GRPC port 8081 inside docker. Keep it 8081 always
tritonserver --model-store triton_server/models --allow-grpc=true --allow-http=false --grpc-port=8081 --allow-metrics=false --allow-gpu-metrics=false &
P1=$!

#Run FAPI Server on port 8080. Should be 8080 always
python3 server.py 8080 &
P2=$!
wait $P1 $P2
