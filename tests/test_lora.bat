@echo off

set LORA=/../../../Models/stable-diffusion/loras/WorldofOrigami.safetensors

python ../src/mdx.py -st 10 -g 5 -s 38473746532 -f lora_scale_01 -ls 0.1 -l %LORA%
python ../src/mdx.py -st 10 -g 5 -s 38473746532 -f lora_scale_05 -ls 0.5 -l %LORA%
python ../src/mdx.py -st 10 -g 5 -s 38473746532 -f lora_scale_10 -ls 1.0 -l %LORA%
