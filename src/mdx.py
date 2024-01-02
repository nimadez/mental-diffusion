#
#   Dec 2023
#   @nimadez
#
#   Mental Diffusion Core
#
__import__('colorama').init(autoreset=True)
__import__('warnings').filterwarnings("ignore", category=UserWarning) # disable esrgan/torchvision warnings

import os
import sys
import gc
import json
import math
import random
import logging
import asyncio
import threading
import websockets
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from colorama import Fore
from http import HTTPStatus
from urllib.request import urlopen
from argparse import ArgumentParser
from realesrgan import RealESRGANer
from PIL.PngImagePlugin import PngInfo
from base64 import b64encode, b64decode
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.image_processor import VaeImageProcessor
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
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
METADATAKEY = "mental-diffusion"


class Context():
    def __init__(self):
        self.hf_cache_exist = True
        self.configs = []

        self.cpu = False
        self.device = "cpu"
        self.dtype = None

        self.pipe = None

        self.model = "sd"
        self.checkpoint = None
        self.upscaler = None

        self.vae = None
        self.lora = None
        self.lora_scale = 1.0

        self.scheduler = ""
        self.prompt = ""
        self.negative = ""
        self.width = 512
        self.height = 512
        self.seed = -1
        self.steps = 25
        self.guidance = 8.0
        self.strength = 1.0
        
        self.image_init = None
        self.image_mask = None

        self.upscale = False
        self.savefile = True
        self.onefile = False
        self.outpath = ".output"
        self.filename = "img"
        self.batch = 1

        self.last_checkpoint = None
        self.pipe_vae = None

        self.text_embeds = []
        self.vae_processor = None
        self.tae = None

ctx = Context()


def load_configs():
    with open(ROOT + "/config.json", "r") as f:
        ctx.configs = json.loads(f.read())

    if ctx.configs["http_proxy"]:
        os.environ['http_proxy'] = ctx.configs["http_proxy"]
        os.environ['https_proxy'] = ctx.configs["http_proxy"]

    ctx.checkpoint = ctx.configs["checkpoints"][0]
    ctx.upscaler = ctx.configs["upscalers"][0]

    ctx.model = ctx.configs["model"]
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
        '-mod', '--model',
        type = str,
        default = ctx.model,
        required = False,
        help = "sd/xl, define model type, (def: config.json)"
    )
    parser.add_argument(
        '-c', '--checkpoint',
        type = str,
        default = ctx.checkpoint,
        required = False,
        help = "set checkpoint by file name or path (def: config.json)"
    )
    parser.add_argument(
        '-u', '--upscaler',
        type = str,
        default = ctx.upscaler,
        required = False,
        help = "optional realesrgan by file name or path (def: config.json)"
    )
    parser.add_argument(
        '-v', '--vae',
        type = str,
        default = ctx.vae,
        required = False,
        help = "optional vae model by file name or path (def: none)"
    )
    parser.add_argument(
        '-l', '--lora',
        type = str,
        default = ctx.lora,
        help = "optional lora model by file name or path (def: None)"
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
        help = "ddpm, ddim, pndm, lms, euler, euler_anc (def: config.json)"
    )
    parser.add_argument(
        '-p', '--prompt',
        type = str,
        default = "portrait of a robot astronaut, horizon zero dawn machine, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, 8k",
        help = "positive prompt text input (def: sample)"
    )
    parser.add_argument(
        '-n', '--negative',
        type = str,
        default = "deformed, distorted, disfigured, poorly drawn, poorly drawn hands, poorly drawn face, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, malformed hands, out of focus, text, watermark",
        help = "negative prompt text input (def: sample)"
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
        help = "steps from 1 to 100 (def: 25)"
    )
    parser.add_argument(
        '-g', '--guidance',
        type = float,
        default = ctx.guidance,
        help = "guidance scale, how closely linked to the prompt (def: 8.0)"
    )
    parser.add_argument(
        '-sr', '--strength',
        type = float,
        default = ctx.strength,
        help = "how much respect the final image should pay to the original (def: 1.0)"
    )
    parser.add_argument(
        '-i', '--image',
        type = str,
        default = ctx.image_init,
        help = "PNG file path or base64 PNG (def: None)"
    )
    parser.add_argument(
        '-m', '--mask',
        type = str,
        default = ctx.image_mask,
        help = "PNG file path or base64 PNG (def: None)"
    )
    parser.add_argument(
        '-up', '--upscale',
        type = lambda x: (str(x).lower() == 'true'),
        default = ctx.upscale,
        help = "true/false, upscale using realesrgan 4x (def: false)"
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
        '-serv', '--server',
        type = int,
        default = -1,
        help = "start websockets server (port is required)"
    )
    parser.add_argument(
        '-meta', '--metadata',
        type = str,
        default = None,
        help = "/path-to-image.png, extract metadata from PNG"
    )
    parser.add_argument(
        '-up4x', '--upscale4x',
        type = str,
        default = None,
        help = "/path-to-image.png, upscale a PNG"
    )
    return parser.parse_args(args)


# Utils


def path_checker():
    if not os.path.exists(ctx.checkpoint):
        print('Error: invalid checkpoint path')
        return False

    if bool(ctx.upscale) and not os.path.exists(ctx.upscaler):
        print('Error: invalid upscaler path')
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
        img = base64Decode(uri)
        return img.convert("RGB")
    elif uri.endswith('.png') or uri.endswith('.PNG'):
        img = Image.open(uri)
        return img.convert("RGB")
    return None


def get_metadata(uri, is_base64=False):
    if is_base64 and uri.startswith('data:image/png'):
        return base64Decode(uri).info
    else:
        if os.path.exists(uri):
            return Image.open(uri).info

    print("Error: unable to read metadata.")
    return None


def hf_cache_check(model, filename):
    cache = try_to_load_from_cache(model, filename=filename)
    if isinstance(cache, str):
        return True
    elif cache is _CACHED_NO_EXIST:
        return False
    else:
        return False


def base64Encode(img, format="png"):
    buffered = BytesIO()
    img.save(buffered, format=format)
    buffered.seek(0)
    b64str = f"data:image/{ format };base64," + b64encode(buffered.getvalue()).decode()
    del buffered
    return b64str


def base64Decode(str):
    return Image.open(BytesIO(b64decode(str.split(',')[1])))


def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# Loaders


def load_checkpoint():
    print("loading checkpoint:", os.path.basename(ctx.checkpoint))
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    if ctx.model == "sd":
        ctx.pipe = StableDiffusionPipeline.from_single_file(
            ctx.checkpoint,
            torch_dtype = ctx.dtype,
            revision = "fp16",
            prediction_type = "epsilon",
            image_size = 512,
            local_files_only = ctx.hf_cache_exist,
            use_safetensors = True,
            force_download = False,
            resume_download = True,
            load_safety_checker = False)
    elif ctx.model == "xl":
        ctx.pipe = StableDiffusionXLPipeline.from_single_file(
            ctx.checkpoint,
            torch_dtype = ctx.dtype,
            revision = "fp16",
            image_size = 1024,
            local_files_only = ctx.hf_cache_exist,
            use_safetensors = True,
            force_download = False,
            resume_download = True,
            load_safety_checker = False)

    ctx.pipe.requires_safety_checker = False
    ctx.pipe.safety_checker = None
    print("checkpoint loaded.")


def load_vae():
    ctx.pipe.vae = AutoencoderKL.from_single_file(
        ctx.vae,
        torch_dtype = ctx.dtype,
        revision = "fp16",
        local_files_only = ctx.hf_cache_exist,
        use_safetensors = True,
        force_download = False,
        resume_download = True,
        load_safety_checker = False)
    print("load vae:", os.path.basename(ctx.vae))


def load_tae():
    if ctx.model == "sd":
        ctx.tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=ctx.dtype).to(device=ctx.device)
        print("load tae: sd")
    elif ctx.model == "xl":
        ctx.tae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=ctx.dtype).to(device=ctx.device)
        print("load tae: sdxl")
    ctx.tae.enable_slicing()
    ctx.tae.enable_tiling()


def load_lora_weights(adapter_name):
    model_path = os.path.dirname(ctx.lora)
    weight_name = os.path.basename(ctx.lora)
    ctx.pipe.load_lora_weights(model_path, weight_name=weight_name, adapter_name=adapter_name)
    #pipe.set_adapters(["a1", "a2"], adapter_weights=[0.5, 1.0]) # combine
    #fuse_lora() unfuse_lora() get_active_adapters()
    print("load lora:", os.path.basename(ctx.lora))


def load_scheduler():
    match ctx.scheduler:
        case "ddpm":
            ctx.pipe.scheduler = DDPMScheduler.from_config(ctx.pipe.scheduler.config)
        case "ddim":
            ctx.pipe.scheduler = DDIMScheduler.from_config(ctx.pipe.scheduler.config)
        case "pndm":
            ctx.pipe.scheduler = PNDMScheduler.from_config(ctx.pipe.scheduler.config)
        case "lms":
            ctx.pipe.scheduler = LMSDiscreteScheduler.from_config(ctx.pipe.scheduler.config)
        case "euler":
            ctx.pipe.scheduler = EulerDiscreteScheduler.from_config(ctx.pipe.scheduler.config)
        case "euler_anc":
            ctx.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(ctx.pipe.scheduler.config)


def load_text_embeds():
    # encode prompt for latent preview
    if ctx.model == "sd":
        ctx.text_embeds = ctx.pipe.encode_prompt(
            prompt = ctx.prompt,
            device = ctx.device,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = ctx.guidance > 1.0,
            negative_prompt = ctx.negative,
            lora_scale = ctx.lora_scale)
    elif ctx.model == "xl":
        ctx.text_embeds = ctx.pipe.encode_prompt(
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
    if ctx.cpu or ctx.device == "cpu":
        ctx.dtype = None
        ctx.device = "cpu"
    print("device:", ctx.device.upper())

    ctx.vae_processor = VaeImageProcessor(do_normalize=True)


def pipeline_to_device():
    if not ctx.pipe or ctx.checkpoint != ctx.last_checkpoint:
        load_checkpoint()
        ctx.last_checkpoint = ctx.checkpoint
        ctx.pipe_vae = ctx.pipe.vae
        load_tae()

    if ctx.device != "cpu":
        ctx.pipe.enable_vae_slicing()   # sliced VAE decode for larger batches
        ctx.pipe.enable_vae_tiling()    # tiled VAE decode/encode for large images
        #ctx.pipe.enable_model_cpu_offload()
        #ctx.pipe.enable_sequential_cpu_offload()
    
    # vae
    if ctx.vae:
        load_vae()
    else:
        ctx.pipe.vae = ctx.pipe_vae

    ctx.pipe.vae.decoder.mid_block.attentions[0]._use_2_0_attn = True
    ctx.pipe.vae.to(ctx.device, ctx.dtype)
    
    ctx.pipe.to(ctx.device, ctx.dtype)

    # lora
    ctx.pipe.unload_lora_weights()
    ctx.pipe.unet.set_attn_processor(AttnProcessor2_0()) # unload lora (previously)

    if ctx.lora:
        load_lora_weights("default")
    else:
        ctx.pipe.enable_attention_slicing(1)    # low vram usage (auto|max|8)


# Inference


def latent_preview_step(i, unet, scheduler, latents, timesteps):
    if ctx.model == "sd":
        with torch.no_grad():
            noise_pred = unet(latents, timesteps, encoder_hidden_states=ctx.text_embeds[0]).sample

        # not fully compatible with discretes (lms, euler, euler_anc)
        alpha_t = torch.sqrt(scheduler.alphas_cumprod)
        sigma_t = torch.sqrt(1 - scheduler.alphas_cumprod)
        a_t, s_t = alpha_t[int(timesteps.tolist())], sigma_t[int(timesteps.tolist())]
        latents = (latents - s_t * noise_pred) / a_t

    elif ctx.model == "xl":
        pass # TODO: incompatible unet added_cond_kwargs
    
    decoded = ctx.tae.decode(latents).sample
    image = ctx.vae_processor.postprocess(decoded)[0]
    image.save(ROOT + '/preview.jpg') # + i


def callback_on_step_end(pipe, i, ts, callback_kwargs):
    #pipe._interrupt = True
    latent_preview_step(i, pipe.unet, pipe.scheduler, callback_kwargs["latents"], ts)
    return callback_kwargs


def get_images(pipe, generator):
    cross = None
    if ctx.lora:
        cross = { "scale": ctx.lora_scale }

    return pipe(
        image = ctx.image_init,
        mask_image = ctx.image_mask,
        prompt = ctx.prompt,
        negative_prompt = ctx.negative,
        width = ctx.width,
        height = ctx.height,
        num_inference_steps = ctx.steps,
        guidance_scale = ctx.guidance,
        strength = ctx.strength,
        lora_scale = ctx.lora_scale, # applied to all LoRA layers of the text encoder
        num_images_per_prompt = 1,
        eta = 0.0,
        generator = generator,
        latents = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        output_type = "pil", # "str"
        return_dict = True,
        callback_on_step_end = callback_on_step_end,
        cross_attention_kwargs = cross)
        #guidance_rescale = 0.7) #def: 0.7


def inference():
    load_scheduler()
    load_text_embeds()

    if ctx.steps * ctx.strength < 1:
        ctx.steps = math.ceil(1 / max(0.1, ctx.strength))

    seed = ctx.seed
    if seed == -1:
        seed = np.random.randint(9223372036854775, size=3, dtype=np.int64)
        seed = int(random.choice(seed))

    if ctx.image_init:
        ctx.image_init = prepare_input_image(ctx.image_init)

    if ctx.image_mask:
        ctx.image_mask = prepare_input_image(ctx.image_mask)

    ctx.width = round(ctx.width / 8) * 8
    ctx.height = round(ctx.height / 8) * 8
    generator = torch.Generator(ctx.device).manual_seed(seed)
    pipe_name = "undefined"
    pipe = None

    if not ctx.image_init and not ctx.image_mask:
        pipe_name = "txt2img"
        pipe = ctx.pipe

    elif ctx.image_init and not ctx.image_mask:
        pipe_name = "img2img"

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

    elif ctx.image_init and ctx.image_mask:
        pipe_name = "inpaint"

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
        pipe_name = "txt2img"
        pipe = ctx.pipe

    print(f"[{ctx.model.upper()}, {pipe_name}, {ctx.scheduler}, {ctx.steps}, {ctx.guidance}, {ctx.strength}, {seed}]")
    images = get_images(pipe, generator)
    img = images[0][0]

    metadata = PngInfo()
    metadata.add_text(METADATAKEY, json.dumps({
        "model": ctx.model.upper(),
        "pipeline": pipe_name,
        "checkpoint": ctx.checkpoint,
        "upscaler": ctx.upscaler,
        "vae": ctx.vae,
        "lora": ctx.lora,
        "lora_scale": ctx.lora_scale,
        "scheduler": ctx.scheduler,
        "prompt": ctx.prompt,
        "negative": ctx.negative,
        "width": ctx.width,
        "height": ctx.height,
        "seed": seed,
        "steps": ctx.steps,
        "guidance": ctx.guidance,
        "strength": ctx.strength
    }))

    count = 1
    filename = f"{ctx.filename}"
    savepath = ctx.outpath + "/" + filename
    while os.path.exists(savepath + ".png"):
        savepath = ctx.outpath + "/" + filename + "_" + str(count)
        count += 1
        
    if ctx.savefile and not ctx.onefile:
        img.save(f"{savepath}.png", pnginfo=metadata)
        print(Fore.GREEN + f"{savepath}.png")

    out_base64 = base64Encode(img, "png")

    if ctx.upscale:
        img = inference_realesrgan(ctx.upscaler, img)
        if ctx.savefile and not ctx.onefile:
            img.save(f"{savepath}_4x.png", pnginfo=metadata)
            print(Fore.GREEN + f"{savepath}_4x.png")

    if ctx.savefile and ctx.onefile:
        img.save(f"{savepath}.png", pnginfo=metadata)
        print(Fore.GREEN + f"{savepath}.png")

    if not ctx.savefile:
        print(Fore.GREEN + 'sent')

    del img
    del metadata
    return out_base64


def inference_realesrgan(model_path, image):
    print("upscaler:", os.path.basename(model_path)) 
    image = image.convert("RGB")
    image = np.array(image, dtype=np.uint8)[..., ::-1]
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale = 4,
        model_path = model_path,
        dni_weight = None,
        model = model,
        tile = 256, # ~512
        tile_pad = 10,
        pre_pad = 0,
        half = "fp16",
        gpu_id = 0)
    img, _ = upsampler.enhance(image, outscale=4)
    img = img[:, :, ::-1]
    img = Image.fromarray(img)
    return img


def inference_realesrgan_standalone(model_path, image, outpath):
    img = prepare_input_image(image)
    img = inference_realesrgan(model_path, img)
    p = f"{outpath}/upscale4x_{random.randrange(1000, 9999)}.png"
    img.save(p)
    print(Fore.GREEN + p)


def create(doinference, **kwargs):
    ctx.model = kwargs.pop("model", ctx.model)
    ctx.checkpoint = kwargs.pop("checkpoint", ctx.checkpoint)
    ctx.upscaler = kwargs.pop("upscaler", ctx.upscaler)
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
    ctx.upscale = kwargs.pop("upscale", ctx.upscale)
    ctx.savefile = kwargs.pop("savefile", ctx.savefile)
    ctx.onefile = kwargs.pop("onefile", ctx.onefile)
    ctx.outpath = kwargs.pop("outpath", ctx.outpath)
    ctx.filename = kwargs.pop("filename", ctx.filename)
    ctx.batch = kwargs.pop("batch", ctx.batch)

    if not path_checker():
        return False
    
    pipeline_to_device()

    if doinference:
        for _ in range(1, ctx.batch + 1):
            print(Fore.CYAN + f"#{_}")
            inference()
            clear_cache()

    return True


# Websockets server


class EchoServer:
    async def echo(self, client):
        async for ws in client:
            ws = json.loads(ws)
            match ws["key"]:
                case "open":
                    await client.send(json.dumps({
                        "checkpoints": ctx.configs["checkpoints"],
                        "vaes": ctx.configs["vaes"],
                        "loras": ctx.configs["loras"],
                        "upscalers": ctx.configs["upscalers"],
                        "model": ctx.configs["model"],
                        "scheduler": ctx.configs["scheduler"],
                        "width": ctx.configs["width"],
                        "height": ctx.configs["height"]
                    }))
                    log.info("connection open")
                    self.count = 1

                case "create":
                    data = ws["val"]
                    is_ok = create(False,
                        model = data["model"],
                        checkpoint = data["checkpoint"],
                        upscaler = data["upscaler"],
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
                        upscale = data["upscale"],
                        savefile = data["savefile"],
                        onefile = data["onefile"],
                        outpath = data["outpath"],
                        filename = data["filename"],
                        batch = data["batch"])

                    if is_ok:
                        for _ in range(1, ctx.batch + 1):
                            if _ == 1:
                                print(Fore.CYAN + f"#{self.count}")
                            else:
                                print(Fore.CYAN + f"#{self.count} {_}")

                            try:
                                await client.send(inference())
                                clear_cache()
                            except KeyboardInterrupt:
                                await client.send('')
                                log.info("interrupted.")
                    else:
                        await client.send('')

                    self.count += 1

                case "upscale":
                    data = ws["val"]
                    inference_realesrgan_standalone(data["upscaler"], data["uri"], data["outpath"])
                    await client.send('')

                case "metadata":
                    await client.send(str(get_metadata(ws["val"]["uri"], True)))


    async def main(self):
        try:
            MAX_SIZE_BYTES = 2 ** 25 # 33MB
            self.server = await websockets.serve(self.echo,
                "localhost", self.port, max_queue=1, max_size=MAX_SIZE_BYTES)
            log.info("running server at " + Fore.CYAN + f"ws://localhost:{ self.port }")
            log.info("CTRL+C terminate running task")
        except Exception:
            log.error("server is already running!")


    def start(self, port):
        self.port = port
        self.count = 1
        try:
            self.loop = asyncio.get_event_loop()
            self.loop.create_task(self.main())
            self.loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            asyncio.get_event_loop().stop()
            clear_cache()
            sys.exit(0)


# Main


def main(args):
    load_configs()

    args = arg_parser(args)

    if args.metadata != None:
        print(get_metadata(args.metadata))
        sys.exit(0)

    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)

    if args.server != -1:
        EchoServer().start(args.server)
        sys.exit(0)

    if args.upscale4x != None:
        if not os.path.exists(args.upscaler):
            print('Error: invalid upscaler path')
            sys.exit(1)
        inference_realesrgan_standalone(args.upscaler, args.upscale4x, args.outpath)
        sys.exit(0)

    create(True,
        model = args.model,
        checkpoint = args.checkpoint,
        upscaler = args.upscaler,
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
        upscale = args.upscale,
        savefile = args.savefile,
        onefile = args.onefile,
        outpath = args.outpath,
        filename = args.filename,
        batch = args.batch)


if __name__ == '__main__':
    try:
        print()
        print(Fore.MAGENTA + "Mental Diffusion 0.6.9")

        if not hf_cache_check("openai/clip-vit-large-patch14", "config.json"):
            ctx.hf_cache_exist = False
            print("Warning: 'openai/clip-vit-large-patch14' not found and will be downloaded.")
        if not hf_cache_check("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "config.json"):
            ctx.hf_cache_exist = False
            print("Warning: 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' not found and will be downloaded.")

        init()
        main(sys.argv[1:])

    except Exception as ex:
        print('Error:', ex)
    except KeyboardInterrupt:
        pass
    finally:
        clear_cache()
        sys.exit(0)
