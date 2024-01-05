## Mental Diffusion

Stable diffusion command-line interface

```Version 0.7.1 alpha```

[Downloads](https://github.com/nimadez/mental-diffusion/releases)<br>
[Changelog](https://github.com/nimadez/mental-diffusion/blob/main/CHANGELOG.md)<br>
[Known Issues](https://github.com/nimadez/mental-diffusion#known-issues)

> [ComfyUI Bridge for VS Code](https://github.com/nimadez/mental-diffusion/tree/main/comfyui) doesn't get many updates, but you can download it.

<img src="media/splash.jpg">

- Command-line interface
- Websockets server
- Websockets client ```electron```
- SD, SDXL, SDXL-Turbo, VAE, LoRA, TAESD
- Text-to-Image, Image-to-Image, Inpaint
- Latents preview (BMP/WebP)
- Upscaler realesrgan 4x
- Read and write PNG with metadata
- Optimized for low specs
- Support CPU and GPU

```
--help                     show this help message and exit

--upscaler   -u     str    set realesrgan by file name or path (def: config.json)
--checkpoint -c     str    set checkpoint by file name or path (def: config.json)
--vae        -v     str    optional vae by file name or path (def: None)
--lora       -l     str    optional lora by file name or path (def: None)
--lorascale  -ls    float  0.0-1.0, lora scale (def: 1.0)
--scheduler  -sc    str    ddim, ddpm, lcm, pndm, euler_anc, euler, lms (def: config.json)
--prompt     -p     str    positive prompt text input (def: sample)
--negative   -n     str    negative prompt text input (def: empty)
--width      -w     int    width value must be divisible by 8 (def: config.json)
--height     -h     int    height value must be divisible by 8 (def: config.json)
--seed       -s     int    seed number, -1 to randomize (def: -1)
--steps      -st    int    steps from 1 to 100 (def: 25)
--guidance   -g     float  guidance scale, how closely linked to the prompt (def: 8.0)
--strength   -sr    float  how much respect the final image should pay to the original (def: 1.0)
--image      -i     str    PNG file path or base64 PNG (def: None)
--mask       -m     str    PNG file path or base64 PNG (def: None)
--upscale    -up    bool   true/false, auto-upscale using realesrgan 4x (def: false)
--savefile   -sv    bool   true/false, save image to PNG, contain metadata (def: true)
--onefile    -of    bool   true/false, save the final result only (def: false)
--outpath    -o     str    /path-to-directory (def: .output)
--filename   -f     str    filename prefix (no png extension)
--batch      -b     int    enter number of repeats to run in batch (def: 1)
--preview    -pv    bool   stepping is slower with the preview (def: True)

--server     -serv  int    start websockets server (port is required)
--metadata   -meta  str    /path-to-image.png, extract metadata from PNG
--upscale4x  -up4x  str    /path-to-image.png, upscale a PNG

[automatic pipeline switch]
txt2img: when the 'image' is empty, the pipeline switches to txt2img
img2img: when the 'image' is not empty, the pipeline switches to img2img
inpaint: when the 'image' and 'mask' are not empty, the pipeline switches to inpaint

* --server or --batch is recommended because there is no need to reload the checkpoint
* SDXL-Turbo: 512x512, --steps from 1 to 5, set --guidance to 0.0
* To load SDXL on 3.5 GB, you need at least 16 GB memory and virtual-memory paging
```

### Websockets server
<img src="media/server.gif">

### Batch
<img src="media/batch.gif">

### Websockets client
<img src="media/client.jpg">

### Latents Preview
<img src="media/preview.gif">

## Known Issues
```
:: Latents preview is not fully supported for SDXL

:: Stepping is slower with the preview
Reminder: one solution is to set "pipe._guidance_scale" to 0.0 after 40%
```

## History
```
↑ Back to the roots (diffusers)
↑ Ported to VS Code
↑ Switch from Diffusers to ComfyUI
↑ Upgrade from sdkit to Diffusers
↑ Undiff renamed to Mental Diffusion
↑ Undiff started with "sdkit"
↑ Created for my personal use
```

<img src="media/devshot_initial.gif?raw=true">

## License
Code released under the [MIT license](https://github.com/nimadez/mental-diffusion/blob/main/LICENSE).

## Credits
- [Hugging Face](https://huggingface.co/)
- [Diffusers](https://github.com/huggingface/diffusers)
- [PyTorch](https://pytorch.org/)
- [Stability-AI](https://github.com/Stability-AI)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [TAESD](https://github.com/madebyollin/taesd)
- [Electron](https://www.electronjs.org/)

<img src="media/ending.jpg">

<img src="media/extra.jpg">
