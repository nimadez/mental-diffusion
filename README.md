## Mental Diffusion

<img src="media/splash.jpg"><br><sub>*model: zavychromaxl_v80*</sub>

**Fast Stable Diffusion CLI**<br>
Powered by [Diffusers](https://github.com/huggingface/diffusers)<br>
Designed for Linux

| MDX | 0.8.5 |
| ------- | --- |
| Python | 3.11, 3.12 |
| Torch | 2.3.1 +cu121 |
| Diffusers | 0.29.0 |

## Features
- SD, **SDXL**
- Load VAE and LoRA weights
- TAESD latents preview *(image and animation)*
- Txt2Img, Img2Img, Inpaint *(automatic pipeline)*
- Batch image generation, multiple images per prompt
- Read/write PNG metadata, auto-rename files
- CPU, GPU, Low VRAM mode *(automatic on 4GB cards)*
- Lightweight and fast, rewritten in **300** lines
- Proxy, offline mode, minimal downloads
- Free to use, study, modify, and distribute 
- Addons: Real-ESRGAN [upscaler x4 script](https://github.com/nimadez/mental-diffusion/blob/main/src/upscale.py)

> SD3 is currently not supported

## Installation
> - 3GB Python packages (5.2GB extracted)
> - 50MB HuggingFace cache (mostly for TAESD)
> - Make sure you have a swap partition or swap file
```
git clone https://github.com/nimadez/mental-diffusion
cd mental-diffusion
```
Automatic installation: *(debian-based distros)*
```
apt install python3-pip python3-venv
sh install-venv.sh
sh install-bin.sh
```
Manual installation:
```
python3 -m venv ~/.venv/mdx
source ~/.venv/mdx/bin/activate
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r ./requirements.txt
deactivate
```
*(optional)* Install realesrgan for the upscaler x4 script:
```
~/.venv/mdx/bin/python3 -m pip install realesrgan
```

## Arguments
```
~/.venv/mdx/bin/python3 mdx.py --help

--type        -t    str     sd, xl (def: custom)
--checkpoint  -c    str     checkpoint.safetensors (def: custom)
--scheduler   -sc   str     ddim, ddpm, lcm, pndm, eulera, euler, lms (def: custom)
--prompt      -p    str     positive prompt
--negative    -n    str     negative prompt
--width       -w    int     divisible by 8 (def: custom)
--height      -h    int     divisible by 8 (def: custom)
--seed        -s    int     -1 randomize (def: -1)
--steps       -st   int     1 to 100+ (def: 24)
--guidance    -g    float   0 - 20.0+ (def: 8.0)
--strength    -sr   float   0 - 1.0 (def: 1.0)
--lorascale   -ls   float   0 - 1.0 (def: 1.0)
--image       -i    str     image.png
--mask        -m    str     mask.png
--vae         -v    str     vae.safetensors
--lora        -l    str     lora.safetensors
--filename    -f    str     filename prefix without .png extension, add {seed} to be replaced (def: img_{seed})
--number      -no   int     number of images to generate per prompt (def: 1)
--batch       -b    int     number of repeats to run in batch, --seed -1 to randomize
--preview     -pv           image and animation, stepping is slower with preview enabled (def: no preview)
--lowvram     -lv           slower if you have enough VRAM, automatic on 4GB cards (def: no lowvram)
--metadata    -meta str     image.png, extract metadata from png
```
```
Default:    mdx -p "prompt" -st 28 -g 7.5
SD:         mdx -t sd -c ./checkpoint.safetensors -w 512 -h 512
SDXL:       mdx -t xl -c ./checkpoint.safetensors -w 768 -h 768
Img2Img:    mdx -i ./image.png -sr 0.5
Inpaint:    mdx -i ./image.png -m ./mask.png
VAE:        mdx -v ./vae.safetensors
LoRA:       mdx -l ./lora.safetensors -ls 0.5
Filename:   mdx -f img_test_{seed}
Number:     mdx -no 4
Batch:      mdx -b 10
Preview:    mdx -pv
Low VRAM:   mdx -lv
Metadata:   mdx -meta ./example.png
```

## Tips & Tricks
```
Preview, cancel and repeat with higher steps:
mdx -p "prompt" -g 8.0 -st 20 -pv (CTRL+C to cancel)
mdx -p "prompt" -g 8.0 -st 50 -s 827362763262387

Resume with Img2Img pipeline:
mdx -p "prompt" -st 20 -f myimage
mdx -p "prompt" -st 80 -i ~/.mdx/myimage.png -sr 0.15

Generate 40 images in less time:
mdx -p "prompt" -b 10 -no 4

Open preview image in a browser across the LAN:
cd ~/.mdx && python3 -m http.server 8000
$ open http://192.168.x.x:8000/preview.bmp

Download HuggingFace cache in a specific path:
mkdir ~/.hfcache && ln -s ~/.hfcache ~/.cache/huggingface

* Enable OFFLINE if you have already downloaded the huggingface cache
* Preview image saved to ~/.mdx/preview.bmp (update on progress)
* Preview animation saved to ~/.mdx/filename.webp
```

## Tests
|  | SD CPU | SD GPU | SDXL GPU
| --- | :---: | :---: | :---: |
| Txt2Img | &check; | &check; | &check; |
| Img2Img | &check; | &check; | &check; |
| Inpaint | &check; | &check; | &check; |
| VAE | &check; | &check; | &check; |
| LoRA | &check; | &check; | &check; |
| Batch | &check; | &check; | &check; |
| Preview | &check; | &check; | &check; |
| Low VRAM |  | &check; | &check; |
> *Tested on Debian Trixie (testing branch) with nvidia driver 535*

## Previous Experiments
> - [Legacy command-line interface and server](https://github.com/nimadez/mental-diffusion/tree/main/legacy/README.md) (diffusers)
> - [ComfyUI bridge for VS Code extension](https://github.com/nimadez/mental-diffusion/tree/main/comfyui/README.md)

<img src="media/splash_legacy.jpg"><br><sub>*model: OpenDalleV1.1*</sub>

## History
```
↑ Rewritten in 300 lines
↑ Port to Linux
↑ Back to Diffusers
↑ Port to Code (webui)
↑ Change to ComfyUI API (webui)
↑ Created for personal use on Windows OS (diffusers)

"AI will bring us back to the age of terminals."
```

## License
Code released under the [MIT license](https://github.com/nimadez/mental-diffusion/blob/main/LICENSE).

## Credits
- [Hugging Face](https://huggingface.co/)
- [Diffusers](https://github.com/huggingface/diffusers)
- [PyTorch](https://pytorch.org/)
- [Stability-AI](https://github.com/Stability-AI)
- [TAESD](https://github.com/madebyollin/taesd)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
