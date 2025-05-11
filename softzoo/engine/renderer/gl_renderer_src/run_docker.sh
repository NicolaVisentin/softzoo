#!/bin/bash

### Run container and compile
sudo docker run \
  -v ${PWD}:/workspace \
  -v /opt/miniconda3:/opt/miniconda3 \
  --runtime=nvidia \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -it flex:latest bash
