#
# mental-diffusion's core
#
import logging
log = logging.getLogger("mental-diffusion")

import os
import sys
import json
from gc import collect

from numpy import array as np_array
from numpy import uint8 as np_uint8
from PIL.PngImagePlugin import PngInfo
from PIL import Image

from torch import (
    cuda,
    float16,
    Generator
)

from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler
)

from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

import utils


class Context():
    def __init__(self):
        self.version = "0.1.6"
        
        self.use_CPU = 0
        self.use_VAE = 0

        self.device = "cpu"
        self.dtype = None

        self.configs = []
        self.paths = {}
        self.checkpoints = []

        self.checkpoint_name = ""
        self.scheduler_name = ""

        self.pipe = None
        self.pipe_name = ""

        self.gfpgan = None
        self.realesrgan = None

ctx = Context()


def preload(config):
    clear_cache()

    ctx.use_CPU = config["use_CPU"]
    ctx.use_VAE = config["use_VAE"]
    ctx.checkpoint_name = config["checkpoint"]
    ctx.scheduler_name = "euler_anc" # ddpm, ddim, pndm, lms, euler, euler_anc

    ctx.configs = [
        "configs/v1-inference.yaml",
        "configs/v1-inpainting-inference.yaml",
        "configs/x4-upscaling.yaml"
    ]

    ctx.paths = {
        "checkpoints_root": config["checkpoints_root"],
        "vae": config["vae"],
        "gfpgan": config["gfpgan"],
        "realesrgan": config["realesrgan"]
    }

    if not os.path.exists(config["checkpoints_root"]):
        print("No checkpoint directory found, unable to start the mental-diffusion.\nHave you set up the checkpoints_root in config.json file?")
        sys.exit(0)

    for ckpt in os.listdir(config["checkpoints_root"]):
        if ckpt.endswith(".safetensors"):
            ctx.checkpoints.append(os.path.splitext( os.path.basename(ckpt) )[0])

    if len(ctx.checkpoints) == 0:
        print("No checkpoints found, unable to start the mental-diffusion.\nHave you set up the checkpoints path in config.json file?")
        sys.exit(0)

    if not os.path.exists(ctx.paths["checkpoints_root"] + ctx.checkpoint_name + ".safetensors"):
        print("Default checkpoint does not exist, unable to start the mental-diffusion.\nHave you set up the default checkpoint in config.json file?")
        sys.exit(0)

    if ctx.use_VAE == 1 and not os.path.exists(ctx.paths["vae"]):
        print("Custom VAE is enabled, but no VAE found, unable to start the mental-diffusion.\nHave you set up the VAE in config.json file?")
        sys.exit(0)
        
    if not os.path.exists(ctx.paths["gfpgan"]):
        print("No GFPGAN found, unable to start the mental-diffusion.\nHave you set up the GFPGAN in config.json file?")
        sys.exit(0)

    if not os.path.exists(ctx.paths["realesrgan"]):
        print("No Real-ESRGAN found, unable to start the mental-diffusion.\nHave you set up the Real-ESRGAN in config.json file?")
        sys.exit(0)

    device_setup()


def init():
    log.info("initialized.")


def device_setup():
    ctx.dtype = float16
    ctx.device = "cuda" if cuda.is_available() else "cpu"
    if ctx.use_CPU == 1 or ctx.device == "cpu":
        ctx.device = "cpu"
        ctx.dtype = None


def load_checkpoint(name):
    if name == "null": # to use the current checkpoint (for headless)
        name = ctx.checkpoint_name

    if ctx.pipe and ctx.checkpoint_name == name:
        return # update checkpoint or use the current

    log.info(f"loading checkpoint [{ name }] ...")

    if not "inpainting" in name.lower():
        cfg = ctx.configs[0] # v1-inference
    else:
        cfg = ctx.configs[1] # v1-inpainting-inference

    logging.getLogger("diffusers").setLevel(logging.ERROR) # hide safety message
    sys.stdout = open(os.devnull, 'w') # remove "global_step key not found" warning
    ctx.checkpoint_name = name
    ctx.pipe = utils.from_single_file(
        StableDiffusionPipeline,
        ctx.paths["checkpoints_root"] + name + ".safetensors",
        original_config_file = cfg,
        local_files_only = True,
        use_safetensors = True,
        torch_dtype = ctx.dtype,
        prediction_type = "epsilon",    # SD 1.x
        image_size = 512,               # SD 1.x
        force_download = False,
        resume_download = True,
        load_safety_checker = False)
    ctx.pipe.requires_safety_checker = False
    ctx.pipe.safety_checker = None
    ctx.pipe.to(ctx.device, ctx.dtype)
    sys.stdout = sys.__stdout__

    load_vae(cfg)
    pipe_optimizer()
    log.info("checkpoint ready.")


def load_vae(config):
    if ctx.use_VAE == 0: return

    ctx.pipe.vae = utils.load_vae_weights(ctx.paths["vae"], ctx.device, 512, config)
    if ctx.pipe.vae is not None:
        ctx.pipe.vae.to(ctx.device, ctx.dtype)
        log.info("vae loaded")
    else:
        log.error("unable to load vae")


def pipe_optimizer():
    ctx.pipe.unet.set_attn_processor(AttnProcessor2_0())
    ctx.pipe.vae.decoder.mid_block.attentions[0]._use_2_0_attn = True
    if ctx.device != "cpu":
        ctx.pipe.enable_model_cpu_offload()     # memory optimization
        ctx.pipe.enable_attention_slicing(1)    # low vram usage (auto|max|8)
        ctx.pipe.enable_vae_slicing()           # sliced VAE decode for larger batches
        ctx.pipe.enable_vae_tiling()            # tiled VAE decode/encode for large images


def set_scheduler(name):
    match name:
        case "ddpm":
            ctx.pipe.scheduler = DDPMScheduler.from_config(ctx.pipe.scheduler.config)
        case "ddim":
            ctx.pipe.scheduler = DDIMScheduler.from_config(ctx.pipe.scheduler.config)
            ctx.pipe.scheduler.register_to_config(clip_sample=False)
        case "pndm":
            ctx.pipe.scheduler = PNDMScheduler.from_config(ctx.pipe.scheduler.config)
            ctx.pipe.scheduler.register_to_config(skip_prk_steps=True)
        case "lms":
            ctx.pipe.scheduler = LMSDiscreteScheduler.from_config(ctx.pipe.scheduler.config)
        case "euler":
            ctx.pipe.scheduler = EulerDiscreteScheduler.from_config(ctx.pipe.scheduler.config)
        case "euler_anc":
            ctx.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(ctx.pipe.scheduler.config)
    ctx.scheduler_name = name
    log.info(f"scheduler [{name}]")


def inference_stablediffusion(
    prompt, negative,
    width, height,
    seed, steps, guidance, strength,
    initimage, maskimage):

    generator = Generator(ctx.device).manual_seed(seed)
    log.info(f"seed [{ seed }]")

    img = None

    if initimage == "":
        ctx.pipe_name = "txt2img"
        log.info("pipeline [txt2img] ...")
        img = ctx.pipe(
            prompt = prompt,
            width = width,
            height = height,
            num_inference_steps = steps,
            guidance_scale = guidance,
            negative_prompt = negative,
            num_images_per_prompt = 1,
            eta = 0.0, # only applies to DDIMScheduler
            generator = generator,
            latents = None,
            prompt_embeds = None,
            negative_prompt_embeds = None,
            output_type = "pil", # "str"
            return_dict = True,
            callback = None,
            callback_steps = 1,
            cross_attention_kwargs = None,
            guidance_rescale = 0.7)[0][0] #def: 0.7
    elif initimage != "" and maskimage == "":
        ctx.pipe_name = "img2img"
        log.info("pipeline [img2img] ...")
        img = StableDiffusionImg2ImgPipeline(
            vae = ctx.pipe.vae,
            text_encoder = ctx.pipe.text_encoder,
            tokenizer = ctx.pipe.tokenizer,
            unet = ctx.pipe.unet,
            scheduler = ctx.pipe.scheduler,
            safety_checker = None,
            feature_extractor = None,
            requires_safety_checker = False,
            )(
                prompt = prompt,
                image = initimage,
                strength = strength,
                num_inference_steps = steps,
                guidance_scale = guidance,
                negative_prompt = negative,
                num_images_per_prompt = 1,
                eta = 0.0,
                generator = generator,
                prompt_embeds = None,
                negative_prompt_embeds = None,
                output_type = "pil",
                return_dict = True,
                callback = None,
                callback_steps = 1,
                cross_attention_kwargs = None)[0][0]
    elif initimage != "" and maskimage != "":
        ctx.pipe_name = "inpaint"
        log.info("pipeline [inpaint] ...")
        img = StableDiffusionInpaintPipeline(
            vae = ctx.pipe.vae,
            text_encoder = ctx.pipe.text_encoder,
            tokenizer = ctx.pipe.tokenizer,
            unet = ctx.pipe.unet,
            scheduler = ctx.pipe.scheduler,
            safety_checker = None,
            feature_extractor = None,
            requires_safety_checker = False,
            )(
                prompt = prompt,
                image = initimage,
                mask_image = maskimage,
                width = width,
                height = height,
                strength = strength,
                num_inference_steps = steps,
                guidance_scale = guidance,
                negative_prompt = negative,
                num_images_per_prompt = 1,
                eta = 0.0,
                generator = generator,
                latents = None,
                prompt_embeds = None,
                negative_prompt_embeds = None,
                output_type = "pil",
                return_dict = True,
                callback = None,
                callback_steps = 1,
                cross_attention_kwargs = None)[0][0]

    del initimage
    del maskimage
    del generator
    log.info("image generated.")
    return img


def inference_gfpgan(image, width=512, height=512, upscale=2):
    image = image.resize((width, height))
    image = image.convert("RGB")
    image = np_array(image, dtype=np_uint8)[..., ::-1]

    if ctx.gfpgan == None:
        ctx.gfpgan = GFPGANer(
            model_path = ctx.paths["gfpgan"],
            upscale = 1, # def: 2
            arch = "clean",
            channel_multiplier = 2,
            bg_upsampler = None)
        
    _, _, img = ctx.gfpgan.enhance(
            image,
            has_aligned = False,
            only_center_face = False,
            paste_back = True,
            weight = 0.5)

    img = img[:, :, ::-1]
    img = Image.fromarray(img)
    del image
    return img


def inference_realesrgan(image, width=512, height=512, scale=4):
    image = image.resize((width, height))
    image = image.convert("RGB")
    image = np_array(image, dtype=np_uint8)[..., ::-1]

    if ctx.realesrgan == None:
        ctx.realesrgan = RealESRGANer(
            scale = 4, # def netscale
            model_path = ctx.paths["realesrgan"],
            dni_weight = None,
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), # (x4 RRDBNet model)
            tile = 256, # ~512
            tile_pad = 10,
            pre_pad = 0,
            half = "fp16",
            gpu_id = 0)

    img, _ = ctx.realesrgan.enhance(image, outscale=scale)

    img = img[:, :, ::-1]
    img = Image.fromarray(img)
    del image
    return img


def prepare_input_image(img, width, height, label):
    if img.startswith('data:image/png'):
        img = utils.base64Decode(img)
        log.info(f"{ label } image [base64]")
    elif img.endswith('.png') or img.endswith('.PNG'): # filepath
        log.info(f"{ label } image [{ img }]")
        img = Image.open(img)
    else:
        log.error(f"bad { label } image [{ img[:50] }]")
        return None
    img = img.resize((width, height))
    return img.convert("RGB")


def create_image(
    checkpoint, scheduler,
    prompt, negative,
    width, height,
    seed, steps, guidance, strength,
    initimage, maskimage,
    facefix, upscale,
    savefile, onefile, outpath, filename):

    load_checkpoint(checkpoint)
    set_scheduler(scheduler)

    if initimage:
        initimage = prepare_input_image(initimage, int(width), int(height), "init")

    if maskimage:
        maskimage = prepare_input_image(maskimage, int(width), int(height), "mask")
            
    img = inference_stablediffusion(
        prompt, negative,
        int(width), int(height),
        int(seed), int(steps), float(guidance), float(strength),
        initimage, maskimage)

    del initimage
    del maskimage

    metadict = {
        "version": f"MD { ctx.version }",
        "checkpoint": ctx.checkpoint_name,
        "scheduler": ctx.scheduler_name,
        "pipeline": ctx.pipe_name,
        "prompt": prompt,
        "negative": negative,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "strength": strength,
        "image": "null",
        "mask": "null",
        "facefix": facefix,
        "upscale": upscale,
        "savefile": savefile,
        "onefile": onefile,
        "outpath": outpath,
        "filename": filename
    }
    metadata = PngInfo()
    metadata.add_text("MD", json.dumps(metadict))

    count = 1
    savepath = outpath + "/" + filename
    while os.path.exists(savepath + ".png"): # respect existing filenames
        savepath = outpath + "/" + filename + "_" + str(count)
        count += 1

    if savefile and not onefile:
        img.save(f"{ savepath }.png", pnginfo=metadata)
        log.info(f"image saved to { savepath }.png")

    if facefix:
        img = inference_gfpgan(img, width=width, height=height, upscale=1)
        log.info("filter [gfpgan] applied.")
        if savefile and not onefile:
            img.save(f"{ savepath }_ff.png", pnginfo=metadata)
            log.info(f"image saved to { savepath }_ff.png")

    out_base64 = utils.base64Encode(img, "png")

    if upscale:
        img = inference_realesrgan(img, width=width, height=height, scale=4)
        log.info("filter [realesrgan 4x] applied.")
        if savefile and not onefile:
            img.save(f"{ savepath }_4x.png", pnginfo=metadata)
            log.info(f"image saved to { savepath }_4x.png")

    if savefile and onefile:
        img.save(f"{ savepath }.png", pnginfo=metadata)
        log.info(f"image saved to { savepath }.png")

    del img
    del metadata
    clear_cache()
    log.info("done")
    return [ metadict, out_base64 ]


def clear_cache():
    collect()
    if cuda.is_available():
        cuda.empty_cache()
        cuda.ipc_collect()

def clear_pipe():
    if ctx.pipe:
        del ctx.pipe
    del ctx.gfpgan
    del ctx.realesrgan
    clear_cache()
