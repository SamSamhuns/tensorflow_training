# Model Webserver

Serve the trained savedmodel with a FastAPI+uvicorn webserver with triton-server on a HTTP API endpoint.

## Docker Setup (Recommended)

```shell
# build docker
bash scripts/build_server_docker.sh
# run docker
bash scripts/run_server_docker.sh -p EXPOSED_HTTP_PORT -g CUDA_DEVICE_NUM
```

## Local Setup

```shell
# inside a virtual env or conda env
pip install -r requirements_server
# run the webserver
bash scripts/start_servers.sh
```