#
#   May 2023
#   @nimadez
#
#   Mental Diffusion
#
#   1. configs
#   2. arguments
#   3. utils
#   4. loaders
#   5. inferences
#   6. server
#   7. headless
#   8. terminal
#   9. main


import os
import sys
import json

VER = "0.7.8"
ROOT = os.path.dirname(os.path.realpath(__file__))
VENV = sys.prefix + "/bin/python"
TEMP = ROOT + "/temp"
MODELS = ROOT + "/models"
ESRGAN = MODELS + "/realesrgan/"
PREVIEW = TEMP + "/preview.bmp"
PREVIEWANIM = TEMP + "/preview.webp"
METADATAKEY = "mental-diffusion"

configs = None
with open(ROOT + "/config.json", "r") as f:
    configs = json.loads(f.read())
    if configs["low_vram"]:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
    if configs["offline"]:
        os.environ["DISABLE_TELEMETRY"] = "YES"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if configs["http_proxy"]:
        os.environ["http_proxy"] = configs["http_proxy"]
        os.environ["https_proxy"] = configs["http_proxy"]


__import__('warnings').filterwarnings("ignore", category=UserWarning) # disable esrgan/torchvision 0.16 warnings
# torchvision 0.17+ basicsr workaround
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as functional
        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass


import gc
import math
import time
import torch
import random
import curses
import logging
import asyncio
import websockets
import numpy as np
from PIL import Image
from io import BytesIO
from colorama import Fore
from urllib import request
from http import HTTPStatus
from getpass import getuser
from argparse import ArgumentParser
from realesrgan import RealESRGANer
from threading import current_thread
from PIL.PngImagePlugin import PngInfo
from base64 import b64encode, b64decode
from platform import system as platform_system
from basicsr.archs.rrdbnet_arch import RRDBNet
from huggingface_hub import snapshot_download, try_to_load_from_cache, _CACHED_NO_EXIST
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


__import__('colorama').init(autoreset=True)
LOG_FORMAT = "%(asctime)s %(threadName)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%X")
logging.getLogger('websockets.server').setLevel(logging.ERROR)
current_thread().name = "Main"
log = logging.getLogger("mental-diffusion")


class Context():
    def __init__(self):
        self.host = None
        self.port = None

        self.use_cpu = True
        self.device = "cpu"
        self.dtype = None

        self.pipe = None
        self.output = ".output"
        self.onefile = False

        self.model = "sd"
        self.pipeline = "txt2img"
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
        self.base64 = False
        self.filename = "img_{seed}"
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
        self.progress = None
        self.captures = []

        self.realesrgans = []

ctx = Context()


def set_configs():
    ctx.use_cpu = configs["use_cpu"]
    ctx.host = configs["host"]
    ctx.port = configs["port"]
    ctx.model = configs["model"]
    ctx.scheduler = configs["scheduler"]
    ctx.width = configs["width"]
    ctx.height = configs["height"]
    ctx.output = configs["output"]
    ctx.onefile = configs["onefile"]
    ctx.checkpoint = configs["checkpoints"][0]
    ctx.realesrgans = configs["realesrgans"]


# arguments


def arg_parser(args):
    parser = ArgumentParser('mdx.py', add_help=False)
    parser.add_argument(
        '--help',
        action='help',
        help='show this help message and exit')
    parser.add_argument(
        '-serv', '--server',
        action = 'store_true',
        help = "start websockets server (port: config.json)"
    )
    parser.add_argument(
        '-upx4', '--upscaler',
        type = str,
        help = "/path-to-image.png, upscale image x4"
    )
    parser.add_argument(
        '-meta', '--metadata',
        type = str,
        default = None,
        help = "/path-to-image.png, extract metadata from PNG"
    )
    parser.add_argument(
        '-mode', '--model',
        type = str,
        default = ctx.model,
        required = False,
        help = "sd/xl, set checkpoint model type (def: config.json)"
    )
    parser.add_argument(
        '-pipe', '--pipeline',
        type = str,
        default = ctx.pipeline,
        required = False,
        help = "txt2img/img2img/inpaint, define pipeline (def: txt2img)"
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
        help = "steps from 1 to 100+ (def: 20)"
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
        '-64', '--base64',
        action = 'store_true',
        help = "do not save the image to a file, get base64 only"
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
        action = 'store_true',
        help = "stepping is slower with preview enabled"
    )
    parser.add_argument(
        '-up', '--upscale',
        type = str,
        default = ctx.upscale,
        help = "auto-upscale x2, x4, x4anime (def: null)"
    )
    return parser.parse_args(args)


# utils


def hf_cache_check(model, filename):
    cache = try_to_load_from_cache(model, filename=filename)
    if isinstance(cache, str):
        return True
    elif cache is _CACHED_NO_EXIST:
        return False
    else:
        return False


def downloader(url, path, name):
    path = path + name
    if not os.path.exists(path):
        print(f"downloading {name} ...")
        torch.hub.download_url_to_file(url, path, progress=True)


def path_linuxifier(path):
    if "$USER" in path:
        path = path.replace("$USER", getuser())
    if path.startswith("~/"):
        path = path.replace("~/", f"/home/{getuser()}/")
    return path


def path_checker_startup():
    if not hf_cache_check("madebyollin/taesd", "config.json"):
        print("notice: model 'madebyollin/taesd' will be downloaded.")
    if not hf_cache_check("madebyollin/taesdxl", "config.json"):
        print("notice: model 'madebyollin/taesdxl' will be downloaded.")
    if not hf_cache_check("openai/clip-vit-large-patch14", "config.json"):
        print("notice: model 'openai/clip-vit-large-patch14' will be downloaded.")
    if not hf_cache_check("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "config.json"):
        print("notice: model 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' will be downloaded.")

    if not os.path.exists(TEMP):
        os.mkdir(TEMP)
    if not os.path.exists(MODELS):
        os.mkdir(MODELS)
    if not os.path.exists(ESRGAN):
        os.mkdir(ESRGAN)

    if not os.path.exists(PREVIEW):
        Image.new('RGB', (256, 256), (15, 19, 26)).save(PREVIEW, format="BMP")
    if not os.path.exists(PREVIEWANIM):
        Image.new('RGB', (256, 256), (15, 19, 26)).save(PREVIEWANIM)

    ctx.output = path_linuxifier(ctx.output)
    if not os.path.exists(ctx.output):
        os.mkdir(ctx.output)


def path_checker():
    ctx.checkpoint = path_linuxifier(ctx.checkpoint)
    if not os.path.exists(ctx.checkpoint):
        print('Error: invalid checkpoint path')
        return False

    if ctx.vae:
        ctx.vae = path_linuxifier(ctx.vae)
        if not os.path.exists(ctx.vae):
            print('Error: invalid VAE path')
            return False

    if ctx.lora:
        ctx.lora = path_linuxifier(ctx.lora)
        if not os.path.exists(ctx.lora):
            print('Error: invalid LoRA path')
            return False

    if ctx.image_init and not ctx.image_init.startswith('data:image/png'):
        ctx.image_init = path_linuxifier(ctx.image_init)
        if not os.path.exists(ctx.image_init):
            print('Error: invalid image path')
            return False
        
    if ctx.image_mask and not ctx.image_mask.startswith('data:image/png'):
        ctx.image_mask = path_linuxifier(ctx.image_mask)
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


def get_metadata(uri):
    if uri.startswith('data:image/png'):
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def is_server_online(url):
    try:
        request.urlopen(url)
        return True
    except:
        return False


# loaders


def init_device(doprint=True):
    # detect device
    ctx.dtype = torch.float16
    ctx.device = "cuda" if torch.cuda.is_available() else "cpu"
    if ctx.use_cpu or ctx.device == "cpu":
        ctx.device = "cpu"
        ctx.dtype = None
    if doprint:
        print("device:", ctx.device.upper())


def load_checkpoint():
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
    print("checkpoint to device ...")


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


def pipeline_to_device():
    if not ctx.pipe or ctx.checkpoint != ctx.last_checkpoint:
        load_checkpoint()
        ctx.pipe.to(ctx.device, ctx.dtype)
        ctx.last_checkpoint = ctx.checkpoint
        ctx.pipe_vae = ctx.pipe.vae
        print("checkpoint loaded.")

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


def pipeline_selector(image_init, image_mask):
    # txt2img
    pipe = ctx.pipe

    if ctx.pipeline == "img2img":
        if not image_init:
            print("Error: no input image specified for img2img")
            return None

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

    elif ctx.pipeline == "inpaint":
        if not image_init or not image_mask:
            print("Error: no input image/mask specified for inpaint")
            return None

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

    return pipe


# inferences


def latents_preview_step(idx, ts, latents, pipe):
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
    
    latents = 1 / ctx.tae.config.scaling_factor * latents
    decoded = ctx.tae.decode(latents).sample
    image = pipe.image_processor.postprocess(decoded)[0]
    image.save(PREVIEW, format="BMP")
    ctx.captures.append(image)


def callback_on_step_end(pipe, idx, ts, callback_kwargs):
    pipe._interrupt = ctx.interrupt
    ctx.progress = { "step": idx, "timestep": ts.item() }
    
    if ctx.preview and not ctx.interrupt:
        latents = callback_kwargs["latents"]
        with torch.no_grad():
            latents_preview_step(idx, ts, latents, pipe)

    return callback_kwargs


def get_images(pipe, generator, steps, width, height, image_init, image_mask):
    cross = None
    if ctx.lora:
        cross = { "scale": ctx.lora_scale }

    # Fix for SDXL latent preview
    # unet raise error: added_cond_kwargs={'text_embeds','time_ids'}
    pipe.unet.config.addition_embed_type = None

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
        seed = random.randrange(0, sys.maxsize)
        # simplified for maximum compatibility
        # https://github.com/nimadez/mental-diffusion/issues/12
        #seed = np.random.randint(9223372036854775, size=3, dtype=np.int64)
        #seed = int(random.choice(seed))

    if image_init:
        image_init = prepare_input_image(image_init)

    if image_mask:
        image_mask = prepare_input_image(image_mask)

    pipe = pipeline_selector(image_init, image_mask)
    if not pipe:
        return None

    if ctx.pipeline == "img2img" or ctx.pipeline == "inpaint":
        width, height = image_init.size

    print(f"[{ctx.model.upper()}, {ctx.pipeline}, {width}x{height}, {ctx.scheduler}, {steps}, {ctx.guidance}, {ctx.strength}, {seed}]")

    ctx.is_working = True
    ctx.interrupt = False
    ctx.progress = None
    ctx.captures = []

    pipe.scheduler.set_timesteps(steps)
    generator = torch.Generator("cpu").manual_seed(seed)
    img = get_images(pipe, generator, steps, width, height, image_init, image_mask)[0][0]
    
    ctx.is_working = False
    ctx.progress = None
    if ctx.interrupt:
        print("interrupted.")
        if not configs["interrupt_save"]:
            del img
            del pipe
            del generator
            return None

    if ctx.preview:
        ctx.captures.append(img)
        ctx.captures[0].save(PREVIEWANIM, save_all=True, append_images=ctx.captures[1:], lossless=False, quality=100, method=4, exact=False, duration=250, loop=1)
        ctx.captures = []

    metadata = {
        "model": ctx.model.upper(),
        "pipeline": ctx.pipeline,
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
    savepath = ctx.output + "/" + filename
    while os.path.exists(savepath + ".png"):
        savepath = ctx.output + "/" + filename + "_" + str(count)
        count += 1
        
    if not ctx.base64 and not ctx.onefile:
        img.save(f"{savepath}.png", pnginfo=pngInfo)
        print(Fore.GREEN + f"{savepath}.png")

    out_base64 = base64_encode(img, "png")
    imgup = None
    
    if ctx.upscale:
        imgup = inference_realesrgan(ctx.upscale, img)
        if imgup and not ctx.base64 and not ctx.onefile:
            imgup.save(f"{savepath}_{ctx.upscale}.png", pnginfo=pngInfo)
            print(Fore.GREEN + f"{savepath}_{ctx.upscale}.png")

    if imgup and not ctx.base64 and ctx.onefile:
        imgup.save(f"{savepath}.png", pnginfo=pngInfo)
        print(Fore.GREEN + f"{savepath}.png")
    elif not imgup and not ctx.base64 and ctx.onefile:
        img.save(f"{savepath}.png", pnginfo=pngInfo)
        print(Fore.GREEN + f"{savepath}.png")

    if ctx.base64:
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
            downloader(ctx.realesrgans[0], ESRGAN, model_name)
        case "x4":
            model_name = "RealESRGAN_x4plus.pth"
            model_path = ESRGAN + model_name
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            downloader(ctx.realesrgans[1], ESRGAN, model_name)
        case "x4anime":
            model_name = "RealESRGAN_x4plus_anime_6B.pth"
            model_path = ESRGAN + model_name
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            downloader(ctx.realesrgans[2], ESRGAN, model_name)
        case _:
            print(f"notice: '{model_name}' is not a valid upscaler name, using default x2.")
            model_name = "RealESRGAN_x2plus.pth"
            model_path = ESRGAN + model_name
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            downloader(ctx.realesrgans[0], ESRGAN, model_name)
    
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
        gpu_id = None) #0,1,2

    image = image.convert("RGB")
    image = np.array(image, dtype=np.uint8)[..., ::-1]
    img, _ = upsampler.enhance(image, outscale=netscale)
    img = img[:, :, ::-1]
    img = Image.fromarray(img)

    del image
    del model
    del upsampler
    return img


def inference_realesrgan_standalone(model_name, image):
    img = prepare_input_image(image)
    img = inference_realesrgan(model_name, img)
    if img:
        filepath = f"{ctx.output}/upscale_{model_name}_{random.randrange(1000, 9999)}.png"
        img.save(filepath)
        print(Fore.GREEN + filepath)
    del img


def create_new_inference(**kwargs):
    ctx.model = kwargs.pop("model", ctx.model)
    ctx.pipeline = kwargs.pop("pipeline", ctx.pipeline)
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
    ctx.base64 = kwargs.pop("base64", ctx.base64)
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
        inference()
        clear_cache()


# server


class EchoServer:
    def __init__(self):
        if platform_system() == 'Windows': # fix [WinError 10054]
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        self.loop = None
        self.count = 0

    async def echo(self, client):
        if ctx.is_working:
            return
        try:
            async for msg in client:
                msg = json.loads(msg)
                match msg["key"]:
                    case "open":
                        log.info("connection open")
                        self.count = 0

                    case "create":
                        self.count += 1
                        data = msg["val"]
                        is_ok = create_new_inference(
                            model = data["model"],
                            pipeline = data["pipeline"],
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
                            base64 = data["base64"],
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
                            inference_realesrgan_standalone(data["upscale"], data["uri"])
                        except KeyboardInterrupt:
                            print('upscale interrupted.')
                        await client.send('')
                        clear_cache()

                    case _:
                        await client.send('')
                        clear_cache()

                await asyncio.sleep(1)
        except websockets.exceptions.ConnectionClosedOK:
            pass

    async def process_request(self, path, request_headers):
        if "Upgrade" in request_headers:
            return
        
        response_headers = [
            ('Server', 'asyncio websocket server'),
            ('Connection', 'close')]

        if path == '/':
            body = str.encode(self.client(f"http://{ctx.host}:{ctx.port}"))
            response_headers.append(('Content-Type', "text/html"))
            response_headers.append(('Content-Length', str(len(body))))
            return HTTPStatus.OK, response_headers, body
        elif path.startswith('/preview'):
            body = open(PREVIEW, 'rb').read()
            response_headers.append(('Content-Type', "image/bmp"))
            response_headers.append(('Content-Length', str(len(body))))
            return HTTPStatus.OK, response_headers, body
        elif path.startswith('/replay'):
            body = open(PREVIEWANIM, 'rb').read()
            response_headers.append(('Content-Type', "image/webp"))
            response_headers.append(('Content-Length', str(len(body))))
            return HTTPStatus.OK, response_headers, body
        elif path.startswith('/interrupt'):
            ctx.interrupt = True
            return HTTPStatus.OK, response_headers
        elif path.startswith('/progress'):
            body = str.encode(json.dumps(ctx.progress))
            response_headers.append(('Content-Type', "application/json"))
            response_headers.append(('Content-Length', str(len(body))))
            return HTTPStatus.OK, response_headers, body
        elif path.startswith('/config'):
            body = str.encode(json.dumps(configs))
            response_headers.append(('Content-Type', "application/json"))
            response_headers.append(('Content-Length', str(len(body))))
            return HTTPStatus.OK, response_headers, body

    async def main(self):
        try:
            MAX_SIZE_BYTES = 2 ** 25 # 33MB
            server = await websockets.serve(self.echo,
                ctx.host, ctx.port, max_queue=1, max_size=MAX_SIZE_BYTES,
                process_request=self.process_request)
            log.info("running server at " + Fore.CYAN + f"http://{ctx.host}:{ctx.port}")
            log.info("CTRL+C terminate the server")
            await server.wait_closed()
        except:
            print("Error: invalid host or port, server already running.")

    def start(self):
        self.loop = asyncio.new_event_loop()
        try:
            self.loop.run_until_complete(self.main())
            self.loop.run_forever()
        except KeyboardInterrupt:
            log.info('server shutdown.')
        finally:
            self.loop.close()

    def client(self, url):
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
    <title>Preview</title>
    <link rel="icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA2lpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNS1jMDE0IDc5LjE1MTQ4MSwgMjAxMy8wMy8xMy0xMjowOToxNSAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDpjY2FhOTMxYi1hZDA5LTkzNDEtOTcyYy1mYmY1ODE3ZGFiNzQiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6OTExRUYxN0RBQkM5MTFFRTgyQzFGRUJERDI2QzgwODQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6OTExRUYxN0NBQkM5MTFFRTgyQzFGRUJERDI2QzgwODQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIChXaW5kb3dzKSI+IDx4bXBNTTpEZXJpdmVkRnJvbSBzdFJlZjppbnN0YW5jZUlEPSJ4bXAuaWlkOjNDQUFDQTVCNjZFNTExRUVCNjFFQTVCQjE4N0EzMTM0IiBzdFJlZjpkb2N1bWVudElEPSJ4bXAuZGlkOjNDQUFDQTVDNjZFNTExRUVCNjFFQTVCQjE4N0EzMTM0Ii8+IDwvcmRmOkRlc2Ny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">
    <style>
        * {{ margin: 0; padding: 0; }}
        body {{ background: #0f131a; color: slategray; font-size: 14px; font-family: monospace; cursor: default; }}
        button {{ padding: 4px }}
        #container {{ position: fixed; left: 50%; top: 50%; transform: translate(-50%,-50%); text-align: center; }}
        #progress {{ position: fixed; background: red; width: 0%; height: 2px; }}
   </style>
</head>
<body>
    <div id="progress"></div>
    <div id="container">
        <img id="preview"><br>
        <button id="interrupt" onclick="fetch('{url}/interrupt')">Interrupt</button>
        <button id="reload" onclick="window.location.reload()">Reload</button>
    </div>
</body>
<script>
    const imgnull = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAYAAAD0eNT6AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA+NpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNS1jMDE0IDc5LjE1MTQ4MSwgMjAxMy8wMy8xMy0xMjowOToxNSAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgeG1wTU06T3JpZ2luYWxEb2N1bWVudElEPSJ1dWlkOmM2YmQzYTg3LTkzNmUtNGE1ZS05ODBkLTZmODE5Yzg0ZWNlNCIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDpFNTAwRkMzQTMyRDMxMUVFOUJEOUE4RDUxMEIxMTkwMyIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDpFNTAwRkMzOTMyRDMxMUVFOUJEOUE4RDUxMEIxMTkwMyIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ0MgKFdpbmRvd3MpIj4gPHhtcE1NOkRlcml2ZWRGcm9tIHN0UmVmOmluc3RhbmNlSUQ9InhtcC5paWQ6NGJiMGI3OTktNjE1Mi02MTQzLTkwYjYtNTYzZGYyZTY1ODUzIiBzdFJlZjpkb2N1bWVudElEPSJ4bXAuZGlkOkEw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";
    const preview = document.getElementById('preview');
    const progress = document.getElementById('progress');
    let progressMax = 0;
    async function renderLoop() {{
        await fetch("{url}");
        requestAnimationFrame(renderLoop);
        setTimeout(async () => {{
            try {{
                preview.src = "{url}/preview?" + new Date().getTime();
                const res = await fetch("{url}/progress");
                const p = await res.text();
                const json = JSON.parse(p);
                const step = parseInt(json.step);
                const timestep = parseInt(json.timestep);
                if (progressMax == 0) progressMax = timestep;
                progress.style.width = 100 - ~~Math.abs((timestep/progressMax)*100) + "%";
            }} catch (err) {{
                progress.style.width = "0";
            }}
        }}, 1000 / 30);
    }} renderLoop();
    preview.onerror = () => {{ preview.src = imgnull; }}
</script>
</html>
"""


# headless


def run_cmd(args):
    is_ok = create_new_inference(
        model = args.model,
        pipeline = args.pipeline,
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
        base64 = args.base64,
        filename = args.filename,
        batch = args.batch,
        preview = args.preview,
        upscale = args.upscale)

    if is_ok:
        do_inference_cli()


# terminal
# dev note:
#   need to import readline to handle left/right arrow keys on user=input(),
#   but readline mess with terminal window resizing.
#   import readline


class Menu(object):
    def __init__(self, items, stdscreen):
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_MAGENTA, -1)
        curses.init_pair(2, curses.COLOR_CYAN, -1)
        curses.init_pair(3, curses.COLOR_GREEN, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.noecho() # no keystroke echos
        curses.cbreak() # not wait for enter

        self.window = curses.initscr()
        self.window.nodelay(1) # makes getch() non-blocking
        self.window.keypad(1)
        self.items = items
        self.px = 0
        self.py = 0

    def navigate(self, n):
        self.py += n
        if self.py < 0:
            self.py = len(self.items) - 1
        elif self.py >= len(self.items):
            self.py = 0

    def create_screen(self, x, y, w, h):
        self.window.addstr(0, 1, f"Mental Diffusion {VER}", curses.color_pair(1))
        self.window.addstr(h-1, 0, f"(q)quit | {ctx.device} | {ctx.host}:{ctx.port}", curses.color_pair(2))
        self.window.addstr(0, int(w-3), "--", curses.color_pair(3))

        for idx, item in enumerate(self.items):
            if 2 + idx < h: # fit menu to screen height
                if idx == self.py:
                    mode = curses.A_REVERSE
                else:
                    mode = curses.A_NORMAL

                msg = "%d. %s" % (idx, item[0])
                self.window.addstr(2 + idx, 1, msg, mode)

    def display(self):
        self.window.clear()
        
        key = None
        y, x = self.window.getyx()
        h, w = self.window.getmaxyx()

        while (key != ord('q')):
            key = self.window.getch()
            y, x = self.window.getyx()
            h, w = self.window.getmaxyx()

            self.create_screen(x, y, w, h)

            if key in [curses.KEY_ENTER, ord("\n")]:
                if self.py == len(self.items) - 1:
                    break
                else:
                    self.items[self.py][1]()

            elif key == curses.KEY_UP:
                self.navigate(-1)
            elif key == curses.KEY_DOWN:
                self.navigate(1)
            elif key == curses.KEY_LEFT:
                self.navigate(1)
            elif key == curses.KEY_RIGHT:
                self.navigate(-1)
            elif key == curses.KEY_RESIZE:
                self.window.clear()

            self.window.refresh()
            time.sleep(0.03)

        self.window.clear()


class Terminal(object):
    def __init__(self, stdscreen):
        curses.curs_set(0)

        self.screen = stdscreen
        self.last_input = None

        items = [
            ("edit [tests/ws-client.js]", self.item_client_edit),
            ("generate [tests/ws-client.js]", self.item_client_generate),
            ("show preview", self.item_client_preview),
            ("interrupt", self.item_client_interrupt),
            ("exit", "exit")]
        self.submenu_client = Menu(items, self.screen)

        items = [
            ("edit config.json", self.item_edit_configs),
            ("download taesd models", self.item_download_taesd),
            ("download realesrgan models", self.item_download_realesrgans),
            ("check python packages", self.item_null),
            ("reload mdx", self.item_reload),
            ("exit", "exit")]
        self.submenu_options = Menu(items, self.screen)

        items = [
            ("start server", self.item_start_server),
            ("start server new terminal", self.item_start_server_new),
            ("websockets client >", self.submenu_client.display),
            ("open preview image", self.item_open_preview_bmp),
            ("open replay animation", self.item_open_preview_webp),
            ("upscale image x4", self.item_upscale),
            ("extract metadata", self.item_metadata),
            ("options >", self.submenu_options.display),
            ("exit", "exit")]
        self.mainmenu = Menu(items, self.screen)
        self.mainmenu.display()

    def clear(self):
        self.screen.clear()
        self.screen.refresh()

    def shell_mode(self):
        curses.reset_shell_mode()
        self.clear()

    def prog_mode(self):
        time.sleep(0.5)
        curses.endwin()
        curses.curs_set(1) # important
        curses.curs_set(0)
        curses.reset_prog_mode()
        self.clear()

    def item_start_server(self):
        self.shell_mode()
        os.system(f"{VENV} {ROOT}/mdx.py -serv")
        input("press enter to exit ...")
        self.prog_mode()

    def item_start_server_new(self):
        self.shell_mode()
        os.system(f"x-terminal-emulator -e {VENV} {ROOT}/mdx.py -serv &")
        self.prog_mode()

    def item_client_edit(self):
        self.shell_mode()
        os.system('nano tests/ws-client.js')
        self.prog_mode()

    def item_client_generate(self):
        self.shell_mode()
        if is_server_online(f"http://{ctx.host}:{ctx.port}"):
            print("connecting to server ...")
            os.system('node tests/ws-client.js >/dev/null')
        else:
            print("server is offline.")
        self.prog_mode()

    def item_client_preview(self):
        self.shell_mode()
        if is_server_online(f"http://{ctx.host}:{ctx.port}"):
            os.system(f"open http://{ctx.host}:{ctx.port} 2>/dev/null")
        else:
            print("server is offline.")
        self.prog_mode()

    def item_client_interrupt(self):
        self.shell_mode()
        if is_server_online(f"http://{ctx.host}:{ctx.port}/interrupt"):
            print("interrupted.")
        else:
            print("server is offline.")
        self.prog_mode()

    def item_open_preview_bmp(self):
        self.shell_mode()
        if os.path.exists(PREVIEW):
            os.system(f"open {PREVIEW} 2>/dev/null")
        self.prog_mode()

    def item_open_preview_webp(self):
        self.shell_mode()
        if os.path.exists(PREVIEWANIM):
            os.system(f"open {PREVIEWANIM} 2>/dev/null")
        self.prog_mode()

    def item_upscale(self):
        self.shell_mode()
        curses.curs_set(1)
        if self.last_input:
            print(f"previous: {self.last_input}\n")
        else:
            print("(?) leave empty to cancel\n")
        img = input("image path: ")
        if img:
            self.last_input = img
            if os.path.exists(img):
                os.system(f"{VENV} {ROOT}/mdx.py -upx4 {img}")
                input("press enter to exit ...")
            else:
                print("image file does not exists.")
        self.prog_mode()

    def item_metadata(self):
        self.shell_mode()
        curses.curs_set(1)
        if self.last_input:
            print(f"previous: {self.last_input}\n")
        else:
            print("(?) leave empty to cancel\n")
        img = input("PNG path: ")
        if img:
            self.last_input = img
            if os.path.exists(img):
                os.system(f"{VENV} {ROOT}/mdx.py -meta {img}")
                input("press enter to exit ...")
            else:
                print("PNG file does not exists.")
        self.prog_mode()

    def item_download_taesd(self):
        self.shell_mode()
        snapshot_download(repo_id="madebyollin/taesd")
        snapshot_download(repo_id="madebyollin/taesdxl")
        input("taesd has been downloaded.\npress enter to exit ...")
        self.prog_mode()

    def item_download_realesrgans(self):
        self.shell_mode()
        downloader(ctx.realesrgans[0], ESRGAN, "RealESRGAN_x2plus.pth")
        downloader(ctx.realesrgans[1], ESRGAN, "RealESRGAN_x4plus.pth")
        downloader(ctx.realesrgans[2], ESRGAN, "RealESRGAN_x4plus_anime_6B.pth")
        input("realesrgan has been downloaded.\npress enter to exit ...")
        self.prog_mode()

    def item_edit_configs(self):
        self.shell_mode()
        os.system(f"nano {ROOT}/config.json")
        print('preparing to restart the script ...')
        os.execv(sys.executable, [sys.prefix + "/bin/python"] + sys.argv)
        self.prog_mode()

    def item_reload(self):
        self.shell_mode()
        print('preparing to restart the script ...')
        os.execv(sys.executable, [sys.prefix + "/bin/python"] + sys.argv)
        self.prog_mode()

    def item_null(self):
        self.shell_mode()
        print('feature is not available.')
        self.prog_mode()


# main


def main(args):
    args = arg_parser(args)

    if args.metadata != None:
        print(get_metadata(args.metadata))
        clear_cache()
        sys.exit(0)

    path_checker_startup()

    if args.upscaler != None:
        inference_realesrgan_standalone("x4", args.upscaler)
        clear_cache()
        sys.exit(0)
    
    if args.server:
        print(Fore.MAGENTA + f"Mental Diffusion {VER}")
        init_device()
        load_taesd()
        EchoServer().start()
        clear_cache()
        sys.exit(0)

    if len(sys.argv) > 1:
        print(Fore.MAGENTA + f"Mental Diffusion {VER}")
        init_device()
        load_taesd()
        run_cmd(args)
        clear_cache()
        sys.exit(0)
        
    init_device(False)
    curses.wrapper(Terminal)
    os.system('clear')
    clear_cache()
    sys.exit(0)


if __name__ == '__main__':
    try:
        clear_cache()
        set_configs()
        main(sys.argv[1:])

    except Exception as ex:
        print('Error:', ex)
        clear_cache()
        sys.exit(1)
    finally:
        clear_cache()
        sys.exit(0)
