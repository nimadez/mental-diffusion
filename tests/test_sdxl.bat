@echo off

set SDXL=/../../../Models/stable-diffusion/checkpoints/xl/sd_xl_base_1.0.safetensors

:: txt2img
python ../src/mdx.py -w 1024 -h 1024 -st 20 -g 5 -f sdxltxt2img --seed 482987365 -c %SDXL%

:: img2img
python ../src/mdx.py -w 1024 -h 1024 -st 20 -g 5 -sr 0.6 -f sdxlimg2img -i .output/sdxltxt2img.png -c %SDXL%

:: inpaint
python ../src/mdx.py -w 1024 -h 1024 -st 20 -g 5 -sr 1 -f sdxlinpaint -i .output/sdxltxt2img.png -m mask1024.png -c %SDXL%
