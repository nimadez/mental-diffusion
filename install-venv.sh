#!/bin/bash

# for debian-based linux distributions
# $ sudo apt install python3-pip python3-venv

mkdir ~/.venv
python3 -m venv ~/.venv/mdx

~/.venv/mdx/bin/python3 -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
~/.venv/mdx/bin/python3 -m pip install --upgrade -r ./requirements.txt
