@echo off

set VAE=/../../../Models/stable-diffusion/vae/TRCVAE.safetensors

python ../src/mdx.py -st 10 -g 5 -f vae -v %VAE%
