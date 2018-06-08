#!/bin/bash

# install.sh

sudo apt-get -y update

sudo apt-get install -y gcc
sudo apt-get install -y g++
sudo apt-get install -y unzip
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libopenmpi-dev
sudo apt-get install -y libhdf5-dev
sudo apt-get install -y zlib1g-dev
sudo apt-get install -y libnetcdf-dev
sudo apt-get install -y libnetcdf-c++4-dev
sudo apt-get install -y libjsoncpp-dev
sudo apt-get install -y libcppunit-dev

git clone https://github.com/amzn/amazon-dsstne
cd amazon-dsstne

wget https://github.com/NVlabs/cub/archive/1.5.2.zip
unzip 1.5.2.zip
sudo cp -rf cub-1.5.2/cub/ /usr/local/include/

export LD_LIBRARY_PATH="/usr/local/cuda/lib64/"
cd src/amazon/dsstne
/usr/bin/make -j24
export PATH=`pwd`/bin:$PATH

cd ../../../
