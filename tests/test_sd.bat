@echo off

:: txt2img
python ../src/mdx.py -st 10 -g 5 -f sdtxt2img --seed 482987365

:: img2img
python ../src/mdx.py -st 10 -g 5 -sr 0.6 -f sdimg2img -i .output/sdtxt2img.png

:: inpaint
python ../src/mdx.py -st 10 -g 5 -sr 1 -f sdinpaint -i .output/sdtxt2img.png -m mask.png
