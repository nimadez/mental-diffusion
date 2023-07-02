## Mental Diffusion

<img style="width:100%;max-width:1280px" src="https://repository-images.githubusercontent.com/646072414/1468b045-c2e4-42f1-89b3-3ee61647458b">

Stable diffusion headless and web interface<br>
Version 0.1.4 alpha<br>
[Online Web Interface](https://nimadez.github.io/mental-diffusion/webui) | [Changelog](https://github.com/nimadez/mental-diffusion/blob/main/CHANGELOG)

- [Features](https://github.com/nimadez/mental-diffusion#features)
- [Headless](https://github.com/nimadez/mental-diffusion#headless)
- [Web Interface](https://github.com/nimadez/mental-diffusion#web-interface)
- [Installation](https://github.com/nimadez/mental-diffusion#installation)
- [Models](https://github.com/nimadez/mental-diffusion#models)
- [Performance Tips](https://github.com/nimadez/mental-diffusion#performance-tips)
- [Known Issues](https://github.com/nimadez/mental-diffusion#known-issues)
- [Credits](https://github.com/nimadez/mental-diffusion#credits)

## Features
- [x] Accelerated Torch 2.0
- [x] Lightweight layer on top of Diffusers
- [x] Fast startup and render
- [x] Easy to use web interface
- [x] Headless console for experts
- [x] Websockets server
- [x] Safetensors-only checkpoints
- [x] JSON configuration file *(model paths)*
- [x] Schedulers *(ddpm, ddim, pndm, lms, euler, euler_anc)*
- [x] Text to image
- [x] Image to image
- [x] Image inpainting
- [x] Image outpainting
- [x] Face restoration
- [x] Upscaling 4x
- [x] Batch rendering
- [x] PNG metadata
- [x] Automatic pipeline switch
- [x] File and base64 image inputs *(PNG)*
- [x] Autodetect huggingface cache *(auto-offline)*
- [x] No safety checker
- [x] No miners, trackers, and telemetry
- [x] Optimized for affordable hardware
- [x] Optimized for slow internet connections
- [x] Support custom VAE
- [ ] Support LoRA
- [ ] Support ControlNet

## Requirements
- At least 16 GB RAM
- NVIDIA GPU with CUDA compute capability (at least 4 GB memory)

## Headless
<img src="media/console.gif?raw=true">

```
Usage:    headless.py --arg "value"
Example:  headless.py -p "prompt" -out base64
Upscale:  headless.py -sr 0.1 -up true -i "/path-to-image.png" -of true
Metadata: headless.py -meta "/path-to-image.png"
Batch:    headless.py -p "prompt" -st 25 -b 10 -w 1200 -h 400

--help              show this help message and exit
--get       -g      fetch data from server by record id (int)

--batch     -b      number of images to render (def: 1)
--scheduler -sc     ddpm, ddim, pndm, lms, euler, euler_anc (def: euler_anc)
--prompt    -p      positive prompt input
--negative  -n      negative prompt input
--width     -w      image width have to be divisible by 8 (def: 512)
--height    -h      image height have to be divisible by 8 (def: 512)
--seed      -s      seed number, 0 to randomize (def: 0)
--steps     -st     steps from 10 to 50, 20-25 is good enough (def: 20)
--cfg       -c      guidance scale, how closely linked to the prompt (def: 7.5)
--strength  -sr     how much respect the final image should pay to the original (def: 0.5)
--lora      -l      TODO (def: 0.0)
--image     -i      PNG file path or base64 PNG (def: '')
--mask      -m      PNG file path or base64 PNG (def: '')
--facefix   -ff     true/false, face restoration using gfpgan (def: false)
--upscale   -up     true/false, upscale using real-esrgan 4x (def: false)
--savefile  -sv     true/false, save image to PNG, contain metadata (def: true)
--onefile   -of     true/false, save the final result only (def: false)
--outpath   -o      /path-to-directory (def: ./.temp)
--filename  -f      filename prefix (.png extension is not required)

--ckpt      -ckpt   change/reload checkpoint by name
--metadata  -meta   /path-to-image.png, extract metadata from PNG
--out       -out    stdout 'metadata' or 'base64' (def: metadata)

* When the image is not empty, the pipeline switches to image-to-image
* When the image and mask are not empty, the pipeline switches to inpainting
* Check server.log for previous records
```
<img src="media/badmask.jpg?raw=true"><br>
Incorrect and correct mask image


## Web Interface
> Notice: The web interface is under active development

<img src="media/webui.jpg?raw=true">

- [x] [Online](https://nimadez.github.io/mental-diffusion/webui) and offline webui
- [x] Drag and drop workflow
- [x] Image comparison A/B<br>
    A - Front canvas *(left)*<br>
    B - Background image *(right)*<br>
- [x] Painting canvas *(brush, line, eraser, mask, color picker)*
- [x] Canvas editor *(flip, hue, saturation, brightness, contrast, sepia, invert)*
- [x] Styles editor *(use predefined keywords, they are included in metadata)*
- [x] Guide the AI using prompts, brush strokes and color adjustments
- [x] Quick mask painting
- [x] Generates input and mask images for outpainting
- [x] Autosave prompts
- [x] Autosave PNG with metadata
- [x] Metadata pool *(single or multiple PNG import)*
- [x] Bake canvas to image
- [x] Pan and zoom canvas
- [x] Undo/redo for painting tools *(brush, line, eraser, mask)*

> - Your data is safe and can be loaded again as long as "Autosave File" is checked<br>
> - If you want your painting to combine with the image pixels, you need to bake the canvas
> - To start mask painting, check "Paint Mask", choose brush or line tool
> - To create outpainting, set "Outpaint Padding" size, your initial image and mask will be generated for you (set Strength to 1.0)

<img src="media/schedulers.jpg?raw=true"><br>
> Schedulers, using equal steps, steps is not enough for PNDM/LMS

<img src="media/facefix.jpg?raw=true"><br>
> GFPGAN was applied to the LMS rendering above

<img src="media/guide_paint.jpg?raw=true"><br>
> Painting bloods to guide AI with GFPGAN result

<img src="media/inpainting.jpg?raw=true"><br>
> About 10 inpaint renders, top is the original

<img src="media/outpainting.jpg?raw=true"><br>
> Outpaint examples, padding 128 and 256

<img src="media/outpainting_paintball.jpg?raw=true"><br>
> Outpaint padding 128, use inpainting to cleanup errors

#### Mouse controls
| *Key* | *Action* |
| --- | --- |
| Left Button | drag, draw, select |
| Middle Button | Zoom reset |
| Right Button | Pan canvas |
| Wheel | Zoom canvas in/out |

#### Keyboard shortcuts
| *Key* | *Action* |
| --- | --- |
| Space | Toggle metadata pool |
| D | Drag tool |
| B | Brush tool |
| L | Line tool |
| E | Eraser tool |
| M | Mask painting mode |
| ] | Increase tool size |
| [ | Decrease tool size |
| + | Increase tool opacity |
| - | Decrease tool opacity |
| CTRL + Enter | Render/Generate |
| CTRL + L | Load PNG metadata |
| CTRL + Z | Undo painting |
| CTRL + X | Redo painting |

## Installation
#### [ Automatic Installation ]
```
curl -o md-installer.py https://raw.githubusercontent.com/nimadez/mental-diffusion/main/installer/md-installer.py
```
#### [ Manual Installation ]
- Download [python-3.10.11-embed-amd64.zip](https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip)
```
curl https://bootstrap.pypa.io/get-pip.py -k --ssl-no-revoke -o get-pip.py
python get-pip.py
python -m pip install accelerate==0.20.3
python -m pip install diffusers==0.17.1
python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install transformers==4.30.0
python -m pip install omegaconf==2.3.0
python -m pip install safetensors==0.3.1
python -m pip install realesrgan==0.3.0
python -m pip install gfpgan==1.3.8
python -m pip install websockets

git clone https://github.com/nimadez/mental-diffusion.git
run.bat       -> start server (url: http://localhost:8011)
headless.py   -> use headless if you are familiar with consoles

* edit "config.json" to define model paths
```
#### [ Automatic One-Time Downloads ]
- 200 MB gfpgan weights *(root directory)*
- 1.6 GB openai/clip-vit-large-patch14 *(huggingface cache)*
```
To prevent re-downloading models, add huggingface cache directory to your environment variables
> setx HF_HOME D:\SDK\.cache\huggingface
```

## Models
Checkpoints:<br>
[v1-5-pruned-emaonly.safetensors](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors)<br>
[sd-v1-5-inpainting.ckpt](https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/main/sd-v1-5-inpainting.ckpt)<br>
[Deliberate_v2.safetensors](https://huggingface.co/XpucT/Deliberate/blob/main/Deliberate_v2.safetensors)<br>
[Deliberate-inpainting.safetensors](https://huggingface.co/XpucT/Deliberate/blob/main/Deliberate-inpainting.safetensors)<br>
[dreamlike-diffusion-1.0.safetensors](https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0/blob/main/dreamlike-diffusion-1.0.safetensors)<br>
[dreamlike-photoreal-2.0.safetensors](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0/blob/main/dreamlike-photoreal-2.0.safetensors)<br>
VAE: [vae-ft-mse-840000-ema-pruned.safetensors](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors) *(optional)*<br>
Face Restore: [GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth) *(required)*<br>
Upscale: [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) *(required)*<br>
- All **.ckpt** checkpoints converted to **.safetensors** *(security)*
- All checkpoints converted to **fp16** *(smaller size, use [prune.py](https://github.com/nimadez/mental-diffusion/tree/main/scripts/prune.py))*
- All inpainting checkpoints must have "inpainting" in their filename
- Checkpoints are optional, edit **config.json** to setup checkpoints
- VAE is optional but recommended for getting optimal results
- Back to the future, SD v1.x only!

<img src="media/backtothefuture.jpg?raw=true">

> - I do not support any models
> - Visit [Civitai.com](https://civitai.com/) for more SD 1.5 checkpoints

## Performance Tips
```
- Loading a new checkpoint may cause performance to drop for a few seconds
- To achieve maximum performance, set the default startup checkpoint in the config.json file
- Try the lower steps first, you can always move on to the next steps
- Higher Steps and CFG values increase render time
- The Strength amount is relative and depends on the number of Steps, higher values increase render time
```

## Known Issues
```
- Although the diffusers network is offline, it may try to
connect to the internet from time to time when loading the checkpoint.
These are bugs or missing data, close the server and run it again.
```

## History
```
- Mental-diffusion started with "sdkit" and later evolved into diffusers
- Rename project from "undiff" to mental-diffusion
- Created for my personal use
```

## License
Code released under the [MIT license](https://github.com/nimadez/mental-diffusion/blob/main/LICENSE).

## Credits
- [Hugging Face](https://huggingface.co/)
- [Stability-AI](https://stability.ai/)
- [RunwayML](https://runwayml.com/)
- [PyTorch](https://pytorch.org/)
- [Diffusers](https://github.com/huggingface/diffusers)
- [sdkit](https://github.com/easydiffusion)
- [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [prune.py](https://github.com/lopho/stable-diffusion-prune)
- [Civitai](https://civitai.com/)
