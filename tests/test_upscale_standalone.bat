@echo off

python ../src/mdx.py -st 1 -g 1 -f upscalestandalone

python ../src/mdx.py -up4x .output/upscalestandalone.png
