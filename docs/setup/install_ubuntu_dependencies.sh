#! /bin/bash

# Hui Xue
# 7/31/2021
# for the deep leanring crash course

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit
fi

if [ -z "$(cat /etc/lsb-release | grep "Ubuntu 20.04")" ] ; then
  echo "Error: This install script is intended for Ubuntu  20.04 only"
  exit 1
fi

apt update --quiet
DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends --no-install-suggests --yes \
  apt-utils \
  build-essential \
  cpio \
  cython3 \
  gcc-multilib \
  git-core \
  jq \
  libatlas-base-dev \
  libfftw3-dev \
  libfreetype6-dev \
  liblapack-dev \
  liblapacke-dev \
  libopenblas-base \
  libopenblas-dev \
  libpugixml-dev \
  net-tools \
  ninja-build \
  pkg-config \
  python3-dev \
  python3-pip \
  software-properties-common \
  wget \
  nlohmann-json3-dev \
  libboost-all-dev \
  git-lfs

snap install --classic code 

pip3 install -U pip setuptools testresources
DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends --no-install-suggests --yes python3-tk

# Rest of the Python "stuff"
pip3 install Cython matplotlib numpy opencv_python pydicom Pillow pyxb scikit-image scikit-learn scipy sympy tk-tools junitparser pandas seaborn pynvml xsdata onnx onnxruntime six future tqdm
pip3 install wandb