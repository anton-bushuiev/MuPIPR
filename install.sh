#!/bin/bash

# Initialize path to conda.sh
if [ $# -eq 0 ]
then
    conda_sh_path="${HOME}/miniconda3/etc/profile.d/conda.sh"
else
    conda_sh_path=$1
fi

# Create conda environment
. conda_sh_path
conda create -n mupipr python=3.6 -y
conda activate mupipr

# Install required packages
conda install tensorflow-gpu==1.7 -c free -y
conda install cudnn==7.0.5 -y  # Without this you may face the error 
                               # "Check failed: stream->parent()->GetConvolveAlgorithms"
conda install h5py -y
conda install keras-gpu==2.2.4 -y  # Just a wrapper for tf.keras

# Install BiLM
cd bilm-tf/
python setup.py install
cd ..

# Install missing packages
conda install tqdm -y
conda install -c anaconda scikit-learn -y

# cd to the running folder and create necessary directories
cd model/scripts
mkdir -p records
mkdir -p results
