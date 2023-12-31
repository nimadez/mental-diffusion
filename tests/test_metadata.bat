@echo off

python ../src/mdx.py -st 1 -g 1 -f metadata

python ../src/mdx.py -meta .output/metadata.png
