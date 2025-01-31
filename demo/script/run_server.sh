#!/bin/bash

cd ../../

port=12000
echo "port $port"


# Before the Python script finishes, kill the MLflow UI process
kill -9 $(lsof -t -i :$port)
pkill -f gunicorn

#run mlflow server problem
apptainer exec --nv container/eval_image.sif mlflow ui --port $port 

