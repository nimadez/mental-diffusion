## Mental Diffusion

<img src="media/splash.jpg">

**Fast Stable Diffusion CLI**<br>
Powered by [Hugging Face](https://huggingface.co/) & [Diffusers](https://github.com/huggingface/diffusers)<br>
Designed for Linux

| [MDX](https://github.com/nimadez/mental-diffusion/blob/main/src/mdx.py) | 0.9.1 |
| ------- | --- |
| Python | **3.12** - 3.11 |
| Torch | 2.3.1 +cu121 |
| Diffusers | 0.30.0 |

[Changelog](https://github.com/nimadez/mental-diffusion/blob/main/CHANGELOG.md)

## Features
- SD, **SDXL**
- Load VAE and LoRA weights
- Txt2Img, Img2Img, Inpaint *(auto-pipeline)*
- TAESD latents preview *(image and animation)*
- Batch image generation, multiple images per prompt
- Read/write PNG metadata, auto-rename files
- CPU, GPU, Low VRAM mode *(auto mode)*
- Lightweight and fast, rewritten in **300** lines
- Proxy, offline mode, minimal downloads

> SD3 is currently not supported.

## Installation
> - 3GB Python packages (5GB extracted)
> - Automatic HuggingFace cache
> - Make sure you have a swap partition or swap file
```
git clone https://github.com/nimadez/mental-diffusion
cd mental-diffusion

# Automatic installation:
sudo apt install python3-pip python3-venv
sh install-venv.sh

# Manual installation:
python3 -m venv ~/.venv/mdx
source ~/.venv/mdx/bin/activate
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r ./requirements.txt
deactivate
```

## Arguments
```
~/.venv/mdx/bin/python3 mdx.py --help

--type        -t    str     sd, xl (def: custom)
--checkpoint  -c    str     /checkpoint.safetensors (def: custom)
--scheduler   -sc   str     ddim, ddpm, euler, eulera, lcm, lms, pndm (def: custom)
--prompt      -p    str     positive prompt
--negative    -n    str     negative prompt
--width       -w    int     divisible by 8 (def: custom)
--height      -h    int     divisible by 8 (def: custom)
--seed        -s    int     -1 randomize (def: -1)
--steps       -st   int     1 to 100+ (def: 24)
--guidance    -g    float   0 - 20.0+ (def: 8.0)
--strength    -sr   float   0 - 1.0 (def: 1.0)
--lorascale   -ls   float   0 - 1.0 (def: 1.0)
--image       -i    str     /image.png
--mask        -m    str     /mask.png
--vae         -v    str     /vae.safetensors
--lora        -l    str     /lora.safetensors
--filename    -f    str     filename prefix without .png extension, add {seed} to be replaced (def: img_{seed})
--output      -o    str     image and preview output directory (def: custom)
--number      -no   int     number of images to generate per prompt (def: 1)
--batch       -b    int     number of repeats to run in batch, --seed -1 to randomize
--preview     -pv           stepping is slower with preview enabled (def: no preview)
--lowvram     -lv           slower if you have enough VRAM, automatic on 4GB cards (def: no lowvram)
--metadata    -meta str     /image.png, extract metadata from png

[automatic pipeline]
Txt2Img: no --image and no --mask
Img2Img: --image and no --mask
Inpaint: --image and --mask
ERROR:   no --image and --mask
```
```
Default:    mdx -p "prompt" -st 28 -g 7.5
SD:         mdx -t sd -c /checkpoint.safetensors -w 512 -h 512
SDXL:       mdx -t xl -c /checkpoint.safetensors -w 768 -h 768
Img2Img:    mdx -i /image.png -sr 0.5
Inpaint:    mdx -i /image.png -m ./mask.png
VAE:        mdx -v /vae.safetensors
LoRA:       mdx -l /lora.safetensors -ls 0.5
Filename:   mdx -f img_test_{seed}
Output:     mdx -o /home/user/.mdx
Number:     mdx -no 4
Batch:      mdx -b 10
Preview:    mdx -pv
Low VRAM:   mdx -lv
Metadata:   mdx -meta ./image.png
```

## Addons
| Addon | Description |
| --- | --- |
| [mdx-caption.py](https://github.com/nimadez/mental-diffusion/blob/main/src/addons/mdx-caption.py) | Transformers image captioning script |
| [mdx-detect.py](https://github.com/nimadez/mental-diffusion/blob/main/src/addons/mdx-detect.py) | Transformers object detection script |
| [mdx-outpaint.py](https://github.com/nimadez/mental-diffusion/blob/main/src/addons/mdx-outpaint.py) | Create outpaint image and mask for inpaint |
| [mdx-upscale.py](https://github.com/nimadez/mental-diffusion/blob/main/src/addons/mdx-upscale.py) | Real-ESRGAN x2 and x4 script |

> See the comments, the description is given in each file separately.

<img src="media/addon_detect.jpg" height="256"> <img src="media/addon_caption.jpg" height="256">

## Direct Inference
Import MDX class to inference from JSON data

```
from mdx import MDX

data = json.loads(data)
data["prompt"] = "new prompt"

parser = argparse.ArgumentParser()
args = parser.parse_args(namespace=argparse.Namespace(**data))

MDX().main(args)
```
> Inference can be interrupted by creating a file named ".interrupt" in the --output directory.

## Tips & Tricks
```
* Enable OFFLINE if you have already downloaded the huggingface cache
* Enable SAVE_ANIM to save the preview animation to {output}/filename.webp

Preview, cancel, and repeat faster:
mdx -p "prompt" -g 8.0 -st 30 -pv
mdx -p "prompt" -g 8.0 -st 30 -s 827362763262387

Content-aware upscaling: (ImageMagick)
mdx -p "prompt" -st 20 -w 512 -h 512 -f image
magick convert ~/.mdx/image.png -resize 200% ~/.mdx/image_up.png
mdx -p "prompt" -st 20 -i ~/.mdx/image_up.png -sr 0.5

Generate 40 images in less time:
mdx -p "prompt" -b 10 -no 4

Extract images from WebP animation: (ImageMagick)
magick convert image.webp jpg

Explore output directory in a browser across the LAN:
cd ~/.mdx && python3 -m http.server 8000
$ open http://192.168.x.x:8000

Download huggingface cache in a specific path:
mkdir ~/.hfcache && ln -s ~/.hfcache ~/.cache/huggingface
```

## Tests
| 0.9.1 | SD CPU | SD GPU | SDXL GPU |
| --- | :---: | :---: | :---: |
| Txt2Img | &check; | &check; | &check; |
| Img2Img | &check; | &check; | &check; |
| Inpaint | &check; | &check; | &check; |
| VAE | &check; | &check; | &check; |
| LoRA | &check; | &check; | &check; |
| Batch | &check; | &check; | &check; |
| Preview | &check; | &check; | &check; |
| Low VRAM |  | &check; | &check; |

- Debian 13 (Trixie in testing branch)
- Kernel 6.9.12
- Nvidia driver 535

## Previous Experiments
<img src="legacy/media/preview.gif">

> - [Gradio user-interface addons](https://github.com/nimadez/mental-diffusion/tree/main/gradio/README.md) (mdx)
> - [Legacy command-line interface and server](https://github.com/nimadez/mental-diffusion/tree/main/legacy/README.md) (diffusers)
> - [ComfyUI bridge for VS Code extension](https://github.com/nimadez/mental-diffusion/tree/main/comfyui/README.md)

## History
```
↑ Experimental Gradio addons (webui)
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
- [Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [Stability-AI](https://github.com/Stability-AI)
- [TAESD](https://github.com/madebyollin/taesd)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

##### Models
- zavychromaxl_v80
- OpenDalleV1.1
- juggernaut_aftermath
