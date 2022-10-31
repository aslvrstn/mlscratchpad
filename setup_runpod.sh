#! /bin/bash

# Beforehand
# cd /workspace
# git clone https://github.com/aslvrstn/mlscratchpad.git

apt update
# Fix some weirdness on default machines
apt install locales
# Make it a tolerable box
apt install vim less
apt install virtualenv
cd /workspace/mlscratchpad
virtualenv -p python3 venv
source venv/bin/activate
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip install wandb==0.12.1
pip install matplotlib
pip install git+https://github.com/neelnanda-io/Easy-Transformer.git

# Install VSCode extensions (jupyter + python)

# Ideas for fixing wandb login: https://github.com/wandb/wandb/issues/1669
