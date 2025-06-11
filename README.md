# Highway-Sparse-Autoencoder
This is a research project investigating the discoveries on applying Sparse Autoencoder to interpreting Highway Driving Scenarios. 

## 1. Set Up
### 1.1. Working Environment Set Up
1. Set up the conda environment. (Tested environment Ubuntu 22.04)
```
conda create -n highway_env python=3.11
conda activate highway_env
```
2. Clone the main working repository.
```
git clone https://github.com/treeizard/Highway-Sparse-Autoencoder.git
cd Highway-Sparse-Autoencoder
```
3. Clone the Highway Env training environment into the working repository.
```
git clone https://github.com/Farama-Foundation/HighwayEnv
```
4. Run the python setup script.
```
python3 setup.py
```
### 1.2. Training Base Model

