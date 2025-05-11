# The docker file is borrowed from https://github.com/columbia-ai-robotics/dextairity
# FROM nvidia/cuda:11.6.1-devel-ubuntu20.04
FROM nvcr.io/nvidia/cuda:9.2-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND} \
    TZ=Etc/UTC

# Dependencies for glvnd and X11.
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Dependency for EGL
RUN apt update && apt install -y cmake build-essential libgl1-mesa-dev freeglut3-dev libglfw3-dev libgles2-mesa-dev

# Install OpenEXR
RUN apt update && apt install -y openexr

RUN mkdir /workspace/ && cd /workspace/

WORKDIR /workspace