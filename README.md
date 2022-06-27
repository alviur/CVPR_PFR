# Repository for the paper Continually Learning Self-Supervised Representations with Projected Functional Regularization



# Setup conda

```
CUDA_VER=$(cat /usr/local/cuda/version.txt | cut -d' ' -f 3 | cut -c 1-4)
echo $CUDA_VER

# For new cuda varsion file is JSON
# CUDA_VER=11.0

conda update -n base -c defaults conda
conda create -n cvc-class-il python=3.8
conda activate cvc-class-il
conda install -y pytorch torchvision cudatoolkit=$CUDA_VER -c pytorch
conda install -y pytorch-metric-learning -c metric-learning -c pytorch
pip install -U pip
pip install tensorflow-gpu tensorboard
```

# Training

To train the Encoder for feature extraction use the files on script folder.



