@echo off

python ../src/mdx.py -w 512 -h 512 -st 10 -g 8.0 -o .output -f tmp

magick.exe .output/tmp.png -resize 1024x1024 .output/tmp_1024.png

python ../src/mdx.py -st 10 -g 8.0 -sr 0.2 -o .output -f upscalenoesrgan -i .output/tmp_1024.png
