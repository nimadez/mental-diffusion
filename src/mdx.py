#
#   May 2023
#   @nimadez
#
#   Mental Diffusion Core
#
ESRGANS = [ "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" ]

__import__('colorama').init(autoreset=True)
__import__('warnings').filterwarnings("ignore", category=UserWarning) # disable esrgan/torchvision warnings

import gc
import os
import sys
import json
import math
import torch
import random
import logging
import asyncio
import platform
import threading
import websockets
import numpy as np
from io import BytesIO
from colorama import Fore
from urllib import request
from http import HTTPStatus
from argparse import ArgumentParser
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from base64 import b64encode, b64decode
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    DDIMScheduler,
    DDPMScheduler,
    LCMScheduler,
    PNDMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    AutoencoderKL, AutoencoderTiny)


LOG_FORMAT = "%(asctime)s %(threadName)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%X")
logging.getLogger('websockets.server').setLevel(logging.ERROR)
threading.current_thread().name = "Main"
log = logging.getLogger("mental-diffusion")


os.environ['DISABLE_TELEMETRY'] = "YES"
os.environ['HF_HUB_DISABLE_TELEMETRY'] = "1"
os.environ['HF_HUB_OFFLINE'] = "1"
os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = "1"


ROOT = os.path.dirname(os.path.realpath(__file__))
ASSETS = ROOT + "/assets"
MODELS = ROOT + "/models"
ESRGAN = MODELS + "/realesrgan/"
PREVIEW = ASSETS + "/preview.bmp"
PREVIEWANIM = ASSETS + "/preview.webp"
METADATAKEY = "mental-diffusion"


class Context():
    def __init__(self):
        self.configs = []

        self.use_cpu = True
        self.device = "cpu"
        self.dtype = None

        self.pipe = None

        self.model = "sd"
        self.checkpoint = None
        self.vae = None
        self.lora = None
        self.lora_scale = 1.0

        self.scheduler = "ddim"
        self.prompt = ""
        self.negative = ""
        self.width = 512
        self.height = 512
        self.seed = -1
        self.steps = 20
        self.guidance = 8.0
        self.strength = 1.0
        
        self.image_init = None
        self.image_mask = None

        self.savefile = True
        self.onefile = False
        self.outpath = ".output"
        self.filename = "img"

        self.batch = 1
        self.preview = False
        self.upscale = None

        self.last_checkpoint = None
        self.pipe_vae = None
        
        self.taesd = None
        self.taesdxl = None
        self.tae = None

        self.hidden_states = None

        self.is_working = False
        self.interrupt = False
        self.captures = []

ctx = Context()


def set_configs():
    ctx.use_cpu = ctx.configs["use_cpu"]
    ctx.model = ctx.configs["model"]
    ctx.checkpoint = ctx.configs["checkpoints"][0]
    ctx.scheduler = ctx.configs["scheduler"]
    ctx.width = ctx.configs["width"]
    ctx.height = ctx.configs["height"]


def arg_parser(args):
    parser = ArgumentParser('mdx.py', add_help=False)
    parser.add_argument(
        '--help',
        action='help',
        help='show this help message and exit')
    parser.add_argument(
        '-serv', '--server',
        type = int,
        default = -1,
        help = "start websockets server (port is required)"
    )
    parser.add_argument(
        '-mod', '--model',
        type = str,
        default = ctx.model,
        required = False,
        help = "sd/xl, set checkpoint model type (def: config.json)"
    )
    parser.add_argument(
        '-c', '--checkpoint',
        type = str,
        default = ctx.checkpoint,
        required = False,
        help = "checkpoint .safetensors path (def: config.json)"
    )
    parser.add_argument(
        '-v', '--vae',
        type = str,
        default = ctx.vae,
        required = False,
        help = "optional vae .safetensors path (def: null)"
    )
    parser.add_argument(
        '-l', '--lora',
        type = str,
        default = ctx.lora,
        help = "optional lora .safetensors path (def: null)"
    )
    parser.add_argument(
        '-ls', '--lorascale',
        type = float,
        default = ctx.lora_scale,
        help = "0.0-1.0, lora scale (def: 1.0)"
    )
    parser.add_argument(
        '-sc', '--scheduler',
        type = str,
        default = ctx.scheduler,
        help = "ddim, ddpm, lcm, pndm, euler_anc, euler, lms (def: config.json)"
    )
    parser.add_argument(
        '-p', '--prompt',
        type = str,
        default = "portrait of a robot astronaut, horizon zero dawn machine, intricate, elegant, highly detailed, smooth, sharp focus, 8k",
        help = "positive prompt text input (def: sample)"
    )
    parser.add_argument(
        '-n', '--negative',
        type = str,
        default = "",
        help = "negative prompt text input (def: empty)"
    )
    parser.add_argument(
        '-w', '--width',
        type = int,
        default = ctx.width,
        help = "width value must be divisible by 8 (def: config.json)"
    )
    parser.add_argument(
        '-h', '--height',
        type = int,
        default = ctx.height,
        help = "height value must be divisible by 8 (def: config.json)"
    )
    parser.add_argument(
        '-s', '--seed',
        type = int,
        default = ctx.seed,
        help = "seed number, -1 to randomize (def: -1)"
    )
    parser.add_argument(
        '-st', '--steps',
        type = int,
        default = ctx.steps,
        help = "steps from 1 to 100+ (def: 25)"
    )
    parser.add_argument(
        '-g', '--guidance',
        type = float,
        default = ctx.guidance,
        help = "0.0-20.0+, how closely linked to the prompt (def: 8.0)"
    )
    parser.add_argument(
        '-sr', '--strength',
        type = float,
        default = ctx.strength,
        help = "0.0-1.0, how much respect the image should pay to the original (def: 1.0)"
    )
    parser.add_argument(
        '-i', '--image',
        type = str,
        default = ctx.image_init,
        help = "PNG file path or base64 PNG (def: null)"
    )
    parser.add_argument(
        '-m', '--mask',
        type = str,
        default = ctx.image_mask,
        help = "PNG file path or base64 PNG (def: null)"
    )
    parser.add_argument(
        '-sv', '--savefile',
        type = lambda x: (str(x).lower() == 'true'),
        default = ctx.savefile,
        help = "true/false, save image to PNG, contain metadata (def: true)"
    )
    parser.add_argument(
        '-of', '--onefile',
        type = lambda x: (str(x).lower() == 'true'),
        default = ctx.onefile,
        help = "true/false, save the final result only (def: false)"
    )
    parser.add_argument(
        '-o', '--outpath',
        type = str,
        default = ctx.outpath,
        help = "/path-to-directory (def: .output)"
    )
    parser.add_argument(
        '-f', '--filename',
        type = str,
        default = ctx.filename,
        help = "filename prefix (no png extension)"
    )
    parser.add_argument(
        '-b', '--batch',
        type = int,
        default = ctx.batch,
        help = "enter number of repeats to run in batch (def: 1)"
    )
    parser.add_argument(
        '-pv', '--preview',
        type = lambda x: (str(x).lower() == 'true'),
        default = ctx.preview,
        help = "stepping is slower with preview enabled (def: false)"
    )
    parser.add_argument(
        '-up', '--upscale',
        type = str,
        default = ctx.upscale,
        help = "x2, x4, x4anime (def: null)"
    )
    parser.add_argument(
        '-meta', '--metadata',
        type = str,
        default = None,
        help = "/path-to-image.png, extract metadata from PNG"
    )
    return parser.parse_args(args)


# Utils


def hf_cache_check(model, filename):
    cache = try_to_load_from_cache(model, filename=filename)
    if isinstance(cache, str):
        return True
    elif cache is _CACHED_NO_EXIST:
        return False
    else:
        return False


def downloader(url, path, name):
    if os.path.exists(path):
        req = request.Request(url, method='HEAD')
        f = request.urlopen(req)
        if f.status == 200:
            size = int(f.headers['Content-Length'])
            size_local = os.path.getsize(path)
            if size != size_local:
                print(f"downloading {name} ...")
                request.urlretrieve(url, path)
        else:
            print('Network error, check your internet connection.')
    else:
        print(f"downloading {name} ...")
        request.urlretrieve(url, path)


def path_checker_startup():
    if not hf_cache_check("madebyollin/taesd", "config.json"):
        print("Warning: 'madebyollin/taesd' not found and will be downloaded.")
    if not hf_cache_check("madebyollin/taesdxl", "config.json"):
        print("Warning: 'madebyollin/taesdxl' not found and will be downloaded.")
    if not hf_cache_check("openai/clip-vit-large-patch14", "config.json"):
        print("Warning: 'openai/clip-vit-large-patch14' not found and will be downloaded.")
    if not hf_cache_check("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "config.json"):
        print("Warning: 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' not found and will be downloaded.")

    if not os.path.exists(ASSETS):
        os.mkdir(ASSETS)
    if not os.path.exists(MODELS):
        os.mkdir(MODELS)
    if not os.path.exists(ESRGAN):
        os.mkdir(ESRGAN)

    Image.new('RGB', (256, 256), (15, 19, 26)).save(PREVIEW, format="BMP")
    if not os.path.exists(PREVIEWANIM):
        Image.new('RGB', (256, 256), (15, 19, 26)).save(PREVIEWANIM)
            

def path_checker():
    if not os.path.exists(ctx.checkpoint):
        print('Error: invalid checkpoint path')
        return False

    if ctx.vae and not os.path.exists(ctx.vae):
        print('Error: invalid VAE path')
        return False

    if ctx.lora and not os.path.exists(ctx.lora):
        print('Error: invalid LoRA path')
        return False

    if ctx.image_init and not ctx.image_init.startswith('data:image/png'):
        if not os.path.exists(ctx.image_init):
            print('Error: invalid image path')
            return False
        
    if ctx.image_mask and not ctx.image_mask.startswith('data:image/png'):
        if not os.path.exists(ctx.image_mask):
            print('Error: invalid mask path')
            return False

    return True


def prepare_input_image(uri):
    if uri.startswith('data:image/png'):
        img = base64_decode(uri)
        return img.convert("RGB")
    elif uri.endswith('.png') or uri.endswith('.PNG'):
        img = Image.open(uri)
        return img.convert("RGB")
    return None


def get_metadata(uri, is_base64=False):
    if is_base64 and uri.startswith('data:image/png'):
        return base64_decode(uri).info
    else:
        if os.path.exists(uri):
            return Image.open(uri).info

    print("Error: unable to read metadata.")
    return None


def base64_encode(img, format="png"):
    buffered = BytesIO()
    img.save(buffered, format=format)
    buffered.seek(0)
    b64str = f"data:image/{ format };base64," + b64encode(buffered.getvalue()).decode()
    del buffered
    return b64str


def base64_decode(str):
    return Image.open(BytesIO(b64decode(str.split(',')[1])))


def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# Loaders


def load_checkpoint():
    if ctx.model == "xl" and 2000000000 <= os.path.getsize(ctx.checkpoint) <= 6000000000:
        print("Warning: selected --model (xl) does not match checkpoint size <6GB")
    elif ctx.model == "sd" and 6500000000 <= os.path.getsize(ctx.checkpoint) <= 8000000000:
        print("Warning: selected --model (sd) does not match checkpoint size >6GB")

    logging.getLogger("diffusers").setLevel(logging.ERROR)
    print("loading checkpoint:", os.path.basename(ctx.checkpoint))

    if ctx.model == "sd":
        ctx.pipe = StableDiffusionPipeline.from_single_file(
            ctx.checkpoint,
            torch_dtype = ctx.dtype,
            revision = "fp16",
            prediction_type = "epsilon",
            image_size = 512,
            local_files_only = False,
            use_safetensors = True,
            extract_ema = True,
            force_download = False,
            resume_download = True,
            load_safety_checker = False)
    elif ctx.model == "xl":
        ctx.pipe = StableDiffusionXLPipeline.from_single_file(
            ctx.checkpoint,
            torch_dtype = ctx.dtype,
            revision = "fp16",
            image_size = 1024,
            local_files_only = False,
            use_safetensors = True,
            extract_ema = True,
            force_download = False,
            resume_download = True,
            load_safety_checker = False)

    if ctx.device != "cpu":
        ctx.pipe.enable_vae_slicing()   # sliced VAE decode for larger batches
        ctx.pipe.enable_vae_tiling()    # tiled VAE decode/encode for large images
        #ctx.pipe.enable_model_cpu_offload()
        #ctx.pipe.enable_sequential_cpu_offload()

    ctx.pipe.requires_safety_checker = False
    ctx.pipe.safety_checker = None
    print("checkpoint loaded.")


def load_vae():
    ctx.pipe.vae = AutoencoderKL.from_single_file(
        ctx.vae,
        torch_dtype = ctx.dtype,
        revision = "fp16",
        local_files_only = False,
        use_safetensors = True,
        force_download = False,
        resume_download = True,
        load_safety_checker = False)
    print("load vae:", os.path.basename(ctx.vae))


def load_lora_weights(adapter_name):
    model_path = os.path.dirname(ctx.lora)
    weight_name = os.path.basename(ctx.lora)
    ctx.pipe.load_lora_weights(model_path, weight_name=weight_name, adapter_name=adapter_name)
    #pipe.set_adapters(["a1", "a2"], adapter_weights=[0.5, 1.0]) # combine
    #fuse_lora() unfuse_lora() get_active_adapters()
    print("load lora:", os.path.basename(ctx.lora))


def load_taesd():
    ctx.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=ctx.dtype).to(ctx.device, ctx.dtype)
    ctx.taesdxl = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=ctx.dtype).to(ctx.device, ctx.dtype)
    ctx.taesd.enable_slicing()
    ctx.taesd.enable_tiling()
    ctx.taesdxl.enable_slicing()
    ctx.taesdxl.enable_tiling()
    print("taesd loaded.")


def load_scheduler():
    match ctx.scheduler:
        case "ddim":
            ctx.pipe.scheduler = DDIMScheduler.from_config(ctx.pipe.scheduler.config)
        case "ddpm":
            ctx.pipe.scheduler = DDPMScheduler.from_config(ctx.pipe.scheduler.config)
        case "lcm":
            ctx.pipe.scheduler = LCMScheduler.from_config(ctx.pipe.scheduler.config)
        case "pndm":
            ctx.pipe.scheduler = PNDMScheduler.from_config(ctx.pipe.scheduler.config)
        case "euler_anc":
            ctx.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(ctx.pipe.scheduler.config)
        case "euler":
            ctx.pipe.scheduler = EulerDiscreteScheduler.from_config(ctx.pipe.scheduler.config)
        case "lms":
            ctx.pipe.scheduler = LMSDiscreteScheduler.from_config(ctx.pipe.scheduler.config)


def load_hidden_states():
    # encodes the prompt into text encoder hidden states
    if ctx.model == "sd":
        ctx.hidden_states = ctx.pipe.encode_prompt(
            prompt = ctx.prompt,
            device = ctx.device,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = ctx.guidance > 1.0,
            negative_prompt = ctx.negative,
            lora_scale = ctx.lora_scale)
    elif ctx.model == "xl":
        ctx.hidden_states = ctx.pipe.encode_prompt(
            prompt = ctx.prompt,
            prompt_2 = None,
            device = ctx.device,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = ctx.guidance > 1.0,
            negative_prompt = ctx.negative,
            negative_prompt_2 = None,
            lora_scale = ctx.lora_scale)


# Pipelines


def init():
    # detect device
    ctx.dtype = torch.float16
    ctx.device = "cuda" if torch.cuda.is_available() else "cpu"
    if ctx.use_cpu or ctx.device == "cpu":
        ctx.device = "cpu"
        ctx.dtype = None
    print("device:", ctx.device.upper())

    load_taesd()


def pipeline_to_device():
    if not ctx.pipe or ctx.checkpoint != ctx.last_checkpoint:
        load_checkpoint()
        ctx.pipe.to(ctx.device, ctx.dtype)
        ctx.last_checkpoint = ctx.checkpoint
        ctx.pipe_vae = ctx.pipe.vae

        # load taesd
        if ctx.model == "sd":
            ctx.tae = ctx.taesd
        elif ctx.model == "xl":
            ctx.tae = ctx.taesdxl
    
    # load vae
    if ctx.vae:
        load_vae()
    else:
        ctx.pipe.vae = ctx.pipe_vae
    ctx.pipe.vae.decoder.mid_block.attentions[0]._use_2_0_attn = True
    ctx.pipe.vae.to(ctx.device, ctx.dtype)
    
    # load lora
    ctx.pipe.unet.set_attn_processor(AttnProcessor2_0())
    ctx.pipe.unload_lora_weights()
    if ctx.lora:
        load_lora_weights("default")
    else:
        ctx.pipe.enable_attention_slicing(1) # low vram usage (auto|max|8)

    # load scheduler
    load_scheduler()

    # load embeds
    load_hidden_states()


# Inference


def latents_preview_step(idx, ts, latents, pipe):
    if ctx.model == "sd":
        latent_model_input = pipe.scheduler.scale_model_input(latents, ts)
        noise_pred = pipe.unet(latent_model_input, ts, encoder_hidden_states=ctx.hidden_states[0]).sample
        
        if not hasattr(pipe.scheduler, 'sigmas'): # non-discretes
            alpha_t = torch.sqrt(pipe.scheduler.alphas_cumprod)[ts]
            sigma_t = torch.sqrt(1 - pipe.scheduler.alphas_cumprod)[ts]
            latents = (latents - sigma_t * noise_pred) / alpha_t
        else: # discretes
            step_index = (pipe.scheduler.timesteps == ts).nonzero().item()
            sigma_t = pipe.scheduler.sigmas[step_index + 1]
            latents = latents - sigma_t * noise_pred
            #step_index bug: scheduler.step(noise_pred, ts, latents).pred_original_sample
            #bug workaround: scheduler._step_index = step_index
            
    #TODO SDXL: unet raise error: added_cond_kwargs={'text_embeds','time_ids'}
    
    latents = 1 / ctx.tae.config.scaling_factor * latents
    decoded = ctx.tae.decode(latents).sample
    image = pipe.image_processor.postprocess(decoded)[0]
    image.save(PREVIEW, format="BMP")
    ctx.captures.append(image)


def callback_on_step_end(pipe, idx, ts, callback_kwargs):
    pipe._interrupt = ctx.interrupt
    
    if ctx.preview and not ctx.interrupt:
        latents = callback_kwargs["latents"]
        with torch.no_grad():
            latents_preview_step(idx, ts, latents, pipe)

    return callback_kwargs


def get_images(pipe, generator, steps, width, height, image_init, image_mask):
    cross = None
    if ctx.lora:
        cross = { "scale": ctx.lora_scale }

    return pipe(
        image = image_init,
        mask_image = image_mask,
        prompt = ctx.prompt,
        #prompt_2 = ctx.prompt,
        negative_prompt = ctx.negative,
        #negative_prompt_2 = ctx.negative,
        width = width,
        height = height,
        num_inference_steps = steps,
        guidance_scale = ctx.guidance,
        strength = ctx.strength,
        lora_scale = ctx.lora_scale, # applied to all LoRA layers of the text encoder
        num_images_per_prompt = 1,
        eta = 0.0, #def: 0.0 for DDIM only
        generator = generator,
        output_type = "pil",
        return_dict = True,
        cross_attention_kwargs = cross,
        #guidance_rescale = 0.0, #def: 0.0
        callback_on_step_end = callback_on_step_end,
        callback_on_step_end_tensor_inputs = ['latents']) # 'prompt_embeds'


def auto_pipeline(image_init, image_mask):
    name = "undefined"
    pipe = None

    if not image_init and not image_mask:
        name = "txt2img"
        pipe = ctx.pipe

    elif image_init and not image_mask:
        name = "img2img"

        if ctx.model == "sd":
            pipe = StableDiffusionImg2ImgPipeline(
                vae = ctx.pipe.vae,
                text_encoder = ctx.pipe.text_encoder,
                tokenizer = ctx.pipe.tokenizer,
                unet = ctx.pipe.unet,
                scheduler = ctx.pipe.scheduler,
                safety_checker = None,
                feature_extractor = None,
                requires_safety_checker = False)
        elif ctx.model == "xl":
            pipe = StableDiffusionXLImg2ImgPipeline(
                vae = ctx.pipe.vae,
                text_encoder = ctx.pipe.text_encoder,
                text_encoder_2 = ctx.pipe.text_encoder_2,
                tokenizer = ctx.pipe.tokenizer,
                tokenizer_2 = ctx.pipe.tokenizer_2,
                unet = ctx.pipe.unet,
                scheduler = ctx.pipe.scheduler)

    elif image_init and image_mask:
        name = "inpaint"

        if ctx.model == "sd":
            pipe = StableDiffusionInpaintPipeline(
                vae = ctx.pipe.vae,
                text_encoder = ctx.pipe.text_encoder,
                tokenizer = ctx.pipe.tokenizer,
                unet = ctx.pipe.unet,
                scheduler = ctx.pipe.scheduler,
                safety_checker = None,
                feature_extractor = None,
                requires_safety_checker = False)
        elif ctx.model == "xl":
            pipe = StableDiffusionXLInpaintPipeline(
                vae = ctx.pipe.vae,
                text_encoder = ctx.pipe.text_encoder,
                text_encoder_2 = ctx.pipe.text_encoder_2,
                tokenizer = ctx.pipe.tokenizer,
                tokenizer_2 = ctx.pipe.tokenizer_2,
                unet = ctx.pipe.unet,
                scheduler = ctx.pipe.scheduler)
    else:
        name = "txt2img"
        pipe = ctx.pipe

    return [ name, pipe ]


def inference():
    width = ctx.width
    height = ctx.height
    steps = ctx.steps
    seed = ctx.seed
    image_init = ctx.image_init
    image_mask = ctx.image_mask

    width = round(width / 8) * 8
    height = round(height / 8) * 8

    if steps * ctx.strength < 1:
        steps = math.ceil(1 / max(0.1, ctx.strength))

    if seed == -1:
        seed = np.random.randint(9223372036854775, size=3, dtype=np.int64)
        seed = int(random.choice(seed))

    if image_init:
        image_init = prepare_input_image(image_init)

    if image_mask:
        image_mask = prepare_input_image(image_mask)

    pipe = auto_pipeline(image_init, image_mask)

    if pipe[0] == "img2img" or pipe[0] == "inpaint":
        width, height = image_init.size

    print(f"[{ctx.model.upper()}, {pipe[0]}, {width}x{height}, {ctx.scheduler}, {steps}, {ctx.guidance}, {ctx.strength}, {seed}]")

    ctx.is_working = True
    ctx.interrupt = False
    ctx.captures = []

    pipe[1].scheduler.set_timesteps(steps)
    generator = torch.Generator("cpu").manual_seed(seed)
    img = get_images(pipe[1], generator, steps, width, height, image_init, image_mask)[0][0]
    
    ctx.is_working = False
    if ctx.interrupt:
        print("interrupted.")
        del img
        del pipe
        del generator
        return None

    if ctx.captures:
        ctx.captures.append(img)
        ctx.captures[0].save(PREVIEWANIM, save_all=True, append_images=ctx.captures[1:], lossless=False, quality=100, method=4, exact=False, duration=250, loop=1)
        ctx.captures = []

    metadata = {
        "model": ctx.model.upper(),
        "pipeline": pipe[0],
        "checkpoint": ctx.checkpoint,
        "vae": ctx.vae,
        "lora": ctx.lora,
        "lora_scale": ctx.lora_scale,
        "scheduler": ctx.scheduler,
        "prompt": ctx.prompt,
        "negative": ctx.negative,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "guidance": ctx.guidance,
        "strength": ctx.strength
    }
    pngInfo = PngInfo()
    pngInfo.add_text(METADATAKEY, json.dumps(metadata))

    count = 1
    filename = ctx.filename.replace("{seed}", str(seed))
    savepath = ctx.outpath + "/" + filename
    while os.path.exists(savepath + ".png"):
        savepath = ctx.outpath + "/" + filename + "_" + str(count)
        count += 1
        
    if ctx.savefile and not ctx.onefile:
        img.save(f"{savepath}.png", pnginfo=pngInfo)
        print(Fore.GREEN + f"{savepath}.png")

    out_base64 = base64_encode(img, "png")
    imgup = None
    
    if ctx.upscale:
        imgup = inference_realesrgan(ctx.upscale, img)
        if imgup and ctx.savefile and not ctx.onefile:
            imgup.save(f"{savepath}_{ctx.upscale}.png", pnginfo=pngInfo)
            print(Fore.GREEN + f"{savepath}_{ctx.upscale}.png")

    if imgup and ctx.savefile and ctx.onefile:
        imgup.save(f"{savepath}.png", pnginfo=pngInfo)
        print(Fore.GREEN + f"{savepath}.png")
    elif not imgup and ctx.savefile and ctx.onefile:
        img.save(f"{savepath}.png", pnginfo=pngInfo)
        print(Fore.GREEN + f"{savepath}.png")

    if not ctx.savefile:
        print(Fore.GREEN + 'sent')

    del img
    del imgup
    del pipe
    del pngInfo
    del generator
    return [ metadata, out_base64 ]


def inference_realesrgan(model_name, image):
    model_path = None
    model = None
    netscale = 0
    match model_name:
        case "x2":
            model_name = "RealESRGAN_x2plus.pth"
            model_path = ESRGAN + model_name
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            downloader(ESRGANS[0], ESRGAN + model_name, model_name)
        case "x4":
            model_name = "RealESRGAN_x4plus.pth"
            model_path = ESRGAN + model_name
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            downloader(ESRGANS[1], ESRGAN + model_name, model_name)
        case "x4anime":
            model_name = "RealESRGAN_x4plus_anime_6B.pth"
            model_path = ESRGAN + model_name
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            downloader(ESRGANS[2], ESRGAN + model_name, model_name)
    
    print("upscaler:", model_name)

    upsampler = RealESRGANer(
        scale = netscale,
        model_path = model_path,
        dni_weight = None,
        model = model,
        tile = 256, # ~512
        tile_pad = 10,
        pre_pad = 0,
        half = "fp16",
        gpu_id = 0)

    image = image.convert("RGB")
    image = np.array(image, dtype=np.uint8)[..., ::-1]
    img, _ = upsampler.enhance(image, outscale=netscale)
    img = img[:, :, ::-1]
    img = Image.fromarray(img)

    del image
    del model
    del upsampler
    return img


def inference_realesrgan_standalone(model_name, image, outpath):
    img = prepare_input_image(image)
    img = inference_realesrgan(model_name, img)
    if img:
        outpath = f"{outpath}/upscale_{model_name}_{random.randrange(1000, 9999)}.png"
        img.save(outpath)
        print(Fore.GREEN + outpath)
    del img


def create_new_inference(**kwargs):
    ctx.model = kwargs.pop("model", ctx.model)
    ctx.checkpoint = kwargs.pop("checkpoint", ctx.checkpoint)
    ctx.vae = kwargs.pop("vae", ctx.vae)
    ctx.lora = kwargs.pop("lora", ctx.lora)
    ctx.lora_scale = kwargs.pop("lora_scale", ctx.lora_scale)
    ctx.scheduler = kwargs.pop("scheduler", ctx.scheduler)
    ctx.prompt = kwargs.pop("prompt", ctx.prompt)
    ctx.negative = kwargs.pop("negative", ctx.negative)
    ctx.width = kwargs.pop("width", ctx.width)
    ctx.height = kwargs.pop("height", ctx.height)
    ctx.seed = kwargs.pop("seed", ctx.seed)
    ctx.steps = kwargs.pop("steps", ctx.steps)
    ctx.guidance = kwargs.pop("guidance", ctx.guidance)
    ctx.strength = kwargs.pop("strength", ctx.strength)
    ctx.image_init = kwargs.pop("image_init", ctx.image_init)
    ctx.image_mask = kwargs.pop("image_mask", ctx.image_mask)
    ctx.savefile = kwargs.pop("savefile", ctx.savefile)
    ctx.onefile = kwargs.pop("onefile", ctx.onefile)
    ctx.outpath = kwargs.pop("outpath", ctx.outpath)
    ctx.filename = kwargs.pop("filename", ctx.filename)
    ctx.batch = kwargs.pop("batch", ctx.batch)
    ctx.preview = kwargs.pop("preview", ctx.preview)
    ctx.upscale = kwargs.pop("upscale", ctx.upscale)

    if not path_checker():
        return False
    
    pipeline_to_device()
    return True


async def do_inference_server(client, count):
    for _ in range(1, ctx.batch + 1):
        if _ == 1:
            print(Fore.CYAN + f"#{count}")
        else:
            print(Fore.CYAN + f"#{count} {_}")

        data = await asyncio.to_thread(inference)
        if data is not None:
            await client.send(json.dumps({
                "metadata": data[0],
                "base64": data[1]
            }))
        else:
            await client.send('')

        del data
        clear_cache()
        await asyncio.sleep(1)


def do_inference_cli():
    for _ in range(1, ctx.batch + 1):
        print(Fore.CYAN + f"#{_}")
        try:
            inference()
        except KeyboardInterrupt:
            ctx.interrupt = True
        clear_cache()


# Websockets server


class EchoServer:
    def __init__(self):
        if platform.system() == 'Windows': # fix [WinError 10054] on windows
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        self.loop = None
        self.port = 0
        self.count = 0


    async def echo(self, client):
        if ctx.is_working:
            return
        async for msg in client:
            msg = json.loads(msg)
            match msg["key"]:
                case "open":
                    await client.send(json.dumps({
                        "checkpoints": ctx.configs["checkpoints"],
                        "vaes": ctx.configs["vaes"],
                        "loras": ctx.configs["loras"],
                        "model": ctx.configs["model"],
                        "scheduler": ctx.configs["scheduler"],
                        "width": ctx.configs["width"],
                        "height": ctx.configs["height"]
                    }))
                    log.info("connection open")
                    self.count = 0

                case "create":
                    self.count += 1
                    data = msg["val"]
                    is_ok = create_new_inference(
                        model = data["model"],
                        checkpoint = data["checkpoint"],
                        vae = data["vae"],
                        lora = data["lora"],
                        lora_scale = data["lora_scale"],
                        scheduler = data["scheduler"],
                        prompt = data["prompt"],
                        negative = data["negative"],
                        width = data["width"],
                        height = data["height"],
                        seed = data["seed"],
                        steps = data["steps"],
                        guidance = data["guidance"],
                        strength = data["strength"],
                        image_init = data["image_init"],
                        image_mask = data["image_mask"],
                        savefile = data["savefile"],
                        onefile = data["onefile"],
                        outpath = data["outpath"],
                        filename = data["filename"],
                        batch = data["batch"],
                        preview = data["preview"],
                        upscale = data["upscale"])

                    if is_ok:
                        await do_inference_server(client, self.count)
                    else:
                        await client.send('')

                case "upscale":
                    data = msg["val"]
                    try:
                        inference_realesrgan_standalone(data["upscale"], data["uri"], data["outpath"])
                    except KeyboardInterrupt:
                        print('upscale interrupted.')
                    await client.send('')
                    clear_cache()

            await asyncio.sleep(1)


    async def process_request(self, path, request_headers):
        if "Upgrade" in request_headers:
            return
        response_headers = [
            ('Server', 'asyncio websocket server'),
            ('Connection', 'close'),
        ]
        if path == '/':
            body = str.encode("Mental Diffusion only runs on Electron, or Websocket client at ws://localhost:port")
            response_headers.append(('Content-Type', "text/html"))
            response_headers.append(('Content-Length', str(len(body))))
            return HTTPStatus.OK, response_headers, body
        elif path.startswith('/interrupt'):
            ctx.interrupt = True
            return HTTPStatus.OK, response_headers


    async def main(self):
        try:
            MAX_SIZE_BYTES = 2 ** 25 # 33MB
            server = await websockets.serve(self.echo,
                "localhost", self.port, max_queue=1, max_size=MAX_SIZE_BYTES,
                process_request=self.process_request)
            log.info("running server at " + Fore.CYAN + f"ws://localhost:{ self.port }")
            log.info("CTRL+C terminate the server")
            await asyncio.gather(server.wait_closed())
        except Exception:
            log.error("server is already running!")


    def start(self, port):
        self.port = port
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.main())
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.loop.stop()


# Command-line


def run_cmd(args):
    is_ok = create_new_inference(
        model = args.model,
        checkpoint = args.checkpoint,
        vae = args.vae,
        lora = args.lora,
        lora_scale = args.lorascale,
        scheduler = args.scheduler,
        prompt = args.prompt,
        negative = args.negative,
        width = args.width,
        height = args.height,
        seed = args.seed,
        steps = args.steps,
        guidance = args.guidance,
        strength = args.strength,
        image_init = args.image,
        image_mask = args.mask,
        savefile = args.savefile,
        onefile = args.onefile,
        outpath = args.outpath,
        filename = args.filename,
        batch = args.batch,
        preview = args.preview,
        upscale = args.upscale)

    if is_ok:
        do_inference_cli()


# Main


def main(args):
    args = arg_parser(args)

    if args.metadata != None:
        print(get_metadata(args.metadata))
        clear_cache()
        sys.exit(0)

    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)

    if args.server != -1:
        EchoServer().start(args.server)
        clear_cache()
        sys.exit(0)

    run_cmd(args)
    clear_cache()
    sys.exit(0)


if __name__ == '__main__':
    try:
        print(Fore.MAGENTA + "Mental Diffusion 0.7.4")

        with open(ROOT + "/config.json", "r") as f:
            ctx.configs = json.loads(f.read())

        if ctx.configs["http_proxy"]:
            os.environ["http_proxy"] = ctx.configs["http_proxy"]
            os.environ["https_proxy"] = ctx.configs["http_proxy"]

        path_checker_startup()
        set_configs()
        
        init()
        main(sys.argv[1:])

    except Exception as ex:
        print('Error:', ex)
        clear_cache()
        sys.exit(1)
    finally:
        clear_cache()
        sys.exit(0)
