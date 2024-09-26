FROM  nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /build
COPY ./scripts/get_tensorrt_environment_variables.bash get_tensorrt_environment_variables.bash
COPY ./scripts/get_cuda_environment_variables.bash get_cuda_environment_variables.bash
COPY ./scripts/install_latest_cmake.bash install_latest_cmake.bash
COPY ./scripts/install_onnx_runtime.bash install_onnx_runtime.bash
COPY ./scripts/install_apps_dependencies.bash install_apps_dependencies.bash

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        gnupg2 \
        lsb-release \
        build-essential \
        software-properties-common \
        cmake \
        git \
        tmux && \
    bash install_latest_cmake.bash && \
    bash install_onnx_runtime.bash && \
    bash install_apps_dependencies.bash && \
    rm -rf /build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./ /workspace
WORKDIR /workspace
RUN make gpu_apps

ENTRYPOINT ["/bin/bash"]
