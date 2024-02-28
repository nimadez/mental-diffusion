@echo off
del .output /s /q /f  >nul 2>&1

pause

set RES=256
set SEED=18736534
set STEP=10
set GUID=1
set VAE=/../../../Models/stable-diffusion/vae/TRCVAE.safetensors
set LORA=/../../../Models/stable-diffusion/loras/WorldofOrigami.safetensors

echo [txt2img]
python ../src/mdx.py -s %SEED% -w %RES% -h %RES% -st %STEP% -g %GUID% -f 001_txt2img_sd

echo [txt2img vae]
python ../src/mdx.py -s %SEED% -w %RES% -h %RES% -st %STEP% -g %GUID% -f 002_txt2img_vae -v %VAE%

echo [txt2img lora]
python ../src/mdx.py -s %SEED% -w %RES% -h %RES% -st %STEP% -g %GUID% -f 003_txt2img_lora -ls 0.5 -l %LORA%

echo [txt2img upscale]
python ../src/mdx.py -s %SEED% -w %RES% -h %RES% -st %STEP% -g %GUID% -up x4 -f 004_txt2img_upscale

echo [img2img]
python ../src/mdx.py -s %SEED% -st %STEP% -g %GUID% -sr 0.6 -i .output/001_txt2img_sd.png -f 005_img2img_sd

echo [img2img upscale]
python ../src/mdx.py -s %SEED% -st 1 -g 0 -up x2 -i .output/001_txt2img_sd.png -of true -f 006_img2img_upscale

echo [inpaint]
python ../src/mdx.py -s %SEED% -st %STEP% -g %GUID% -sr 1 -i .output/001_txt2img_sd.png -m mask.png -f 007_inpaint_sd

echo [metadata]
python ../src/mdx.py -meta .output/007_inpaint_sd.png

pause
