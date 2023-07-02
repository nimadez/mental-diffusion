#
# mental-diffusion diffusers
#
import logging
log = logging.getLogger("mental-diffusion")

import os
import sys
import json

from numpy import array as np_array
from numpy import uint8 as np_uint8
from PIL.PngImagePlugin import PngInfo
from PIL import Image

from torch import (
    cuda,
    float16,
    Generator
)

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

from realesrgan import RealESRGANer
from gfpgan import GFPGANer

import utils


class Context():
    def __init__(self):
        self.version = "0.1.4"

        self.network = True
        self.device = "cpu"
        self.dtype = None

        self.configs = []
        self.paths = {}
        self.checkpoints = []

        self.checkpoint_name = ""
        self.scheduler_name = ""
        self.pipe_name = ""

        self.pipe = None
        self.vae = None
        self.realesrgan = None
        self.gfpgan = None

        self.use_CPU = 0
        self.use_VAE = 0

ctx = Context()


def preload():
    clear_cache()
    device_setup()
    ctx.network = utils.defineNetwork()

    with open("./config.json", "r") as f:
        config = json.loads(f.read())

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

    for ckpt in os.listdir(config["checkpoints_root"]):
        if ckpt.endswith(".safetensors"):
            ctx.checkpoints.append(os.path.splitext( os.path.basename(ckpt) )[0])

    ctx.checkpoint_name = config["checkpoint"]
    ctx.scheduler_name = "euler_anc" # ddpm, ddim, pndm, lms, euler, euler_anc
    ctx.use_CPU = int(config["use_CPU"])
    ctx.use_VAE = int(config["use_VAE"])
    del config

    if len(ctx.checkpoints) == 0:
        log.error("No checkpoints found, unable to start the mental-diffusion.\nHave you set up the checkpoints path in config.json file?")
        clear_cache()
        sys.exit(0)

    if not os.path.exists(ctx.paths["checkpoints_root"] + ctx.checkpoint_name + ".safetensors"):
        log.error("Default checkpoint does not exist, unable to start the mental-diffusion.\nHave you set up the default checkpoint in config.json file?")
        clear_cache()
        sys.exit(0)

    if ctx.use_VAE == 1 and not os.path.exists(ctx.paths["vae"]):
        log.error("Custom VAE is enabled, but no VAE found, unable to start the mental-diffusion.\nHave you set up the VAE in config.json file?")
        clear_cache()
        sys.exit(0)

    if not os.path.exists(ctx.paths["gfpgan"]):
        log.error("No GFPGAN found, unable to start the mental-diffusion.\nHave you set up the GFPGAN in config.json file?")
        clear_cache()
        sys.exit(0)

    if not os.path.exists(ctx.paths["realesrgan"]):
        log.error("No ESRGAN found, unable to start the mental-diffusion.\nHave you set up the ESRGAN in config.json file?")
        clear_cache()
        sys.exit(0)


def init():
    load_checkpoint(ctx.checkpoint_name)
    set_optimizer()
    checkpoint_to_device()
    load_VAE()
    initFilters()
    log.info("initialized.")


def initFilters():
    from basicsr.archs.rrdbnet_arch import RRDBNet
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
    ctx.gfpgan = GFPGANer(
        model_path = ctx.paths["gfpgan"],
        upscale = 1, # def: 2
        arch = "clean",
        channel_multiplier = 2,
        bg_upsampler = None)
    log.info("filters ready")


def device_setup():
    ctx.dtype = float16
    ctx.device = "cuda" if cuda.is_available() else "cpu"
    if ctx.use_CPU == 1:
        ctx.device = "cpu"
        ctx.dtype = None


def load_checkpoint(name):
    modelpath = ctx.paths["checkpoints_root"] + name + ".safetensors"
    log.info(f"loading checkpoint [{ name }] ...")
    logging.getLogger("diffusers").setLevel(logging.ERROR) # hide safety message

    if not "inpainting" in name.lower():
        cfg = ctx.configs[0] # v1-inference
    else:
        cfg = ctx.configs[1] # v1-inpainting-inference

    ctx.checkpoint_name = name
    ctx.pipe = utils.StableDiffusionPipeline_from_ckpt(
        StableDiffusionPipeline,
        modelpath,
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
    log.info("prepared checkpoint for device")


def update_checkpoint(name):
    if not name in ctx.checkpoints:
        log.error("checkpoint [%s] does not exist", name)
        return

    try:
        clear_pipe()
        load_checkpoint(name)
        set_optimizer()
        checkpoint_to_device()
        if ctx.vae:
            ctx.pipe.vae = ctx.vae
        log.info("checkpoint updated to [%s]", name)
    except:
        log.error("unable to load checkpoint [%s]", name)
    

def checkpoint_to_device():
    ctx.pipe.to(ctx.device, ctx.dtype)
    log.info("checkpoint loaded to device")


def load_VAE(modelconfig = "configs/v1-inference.yaml"):
    if ctx.use_VAE == 0:
        return
        
    ctx.vae = utils.load_VAE_weights(ctx.paths["vae"], ctx.device, 512, modelconfig)
    if ctx.vae:
        ctx.pipe.vae = ctx.vae
        ctx.pipe.vae.to(ctx.device, ctx.dtype)
        log.info("vae loaded to device")
    else:
        log.error("unable to load vae")


def set_optimizer():
    from diffusers.models.attention_processor import AttnProcessor2_0
    ctx.pipe.enable_model_cpu_offload()     # memory optimization
    ctx.pipe.enable_attention_slicing(1)    # low vram usage
    ctx.pipe.enable_vae_slicing()           # using torch 2
    ctx.pipe.vae.decoder.mid_block.attentions[0]._use_2_0_attn = True
    ctx.pipe.unet.set_attn_processor(AttnProcessor2_0())
    log.info("set optimizations")


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


def inference_gfpgan(initimage, width=512, height=512, upscale=2):
    initimage = initimage.resize((width, height))
    initimage = initimage.convert("RGB")
    initimage = np_array(initimage, dtype=np_uint8)[..., ::-1]

    _, _, img = ctx.gfpgan.enhance(
        initimage,
        has_aligned = False,
        only_center_face = False,
        paste_back = True,
        weight = 0.5)

    img = img[:, :, ::-1]
    img = Image.fromarray(img)
    del initimage
    return img


def inference_realesrgan(initimage, width=512, height=512, scale=4):
    initimage = initimage.resize((width, height))
    initimage = initimage.convert("RGB")
    initimage = np_array(initimage, dtype=np_uint8)[..., ::-1]

    img, _ = ctx.realesrgan.enhance(initimage, outscale=scale)

    img = img[:, :, ::-1]
    img = Image.fromarray(img)
    del initimage
    return img


def generate_image(
    prompt, negative,
    width, height,
    seed, steps, cfg, strength,
    lora,
    initimage, imagemask):

    generator = Generator(ctx.device).manual_seed(seed)
    log.info(f"seed [{ seed }]")

    images = []
    pipe_img2img = None
    pipe_inpainting = None

    if initimage == "":
        ctx.pipe_name = "txt2img"
        log.info("using [txt2img] pipeline...")
        images = ctx.pipe(
            prompt = prompt,
            width = width,
            height = height,
            num_inference_steps = steps,
            guidance_scale = cfg,
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
            cross_attention_kwargs = None)
    elif initimage != "" and imagemask == "":
        ctx.pipe_name = "img2img"
        log.info("using [img2img] pipeline...")
        pipe_img2img = StableDiffusionImg2ImgPipeline(
            vae = ctx.pipe.vae,
            text_encoder = ctx.pipe.text_encoder,
            tokenizer = ctx.pipe.tokenizer,
            unet = ctx.pipe.unet,
            scheduler = ctx.pipe.scheduler,
            safety_checker = None,
            feature_extractor = None,
            requires_safety_checker = False,
        )
        images = pipe_img2img(
            prompt = prompt,
            image = initimage,
            strength = strength,
            num_inference_steps = steps,
            guidance_scale = cfg,
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
            cross_attention_kwargs = None)
    elif initimage != "" and imagemask != "":
        ctx.pipe_name = "inpaint"
        log.info("using [inpaint] pipeline...")
        pipe_inpainting = StableDiffusionInpaintPipeline(
            vae = ctx.pipe.vae,
            text_encoder = ctx.pipe.text_encoder,
            tokenizer = ctx.pipe.tokenizer,
            unet = ctx.pipe.unet,
            scheduler = ctx.pipe.scheduler,
            safety_checker = None,
            feature_extractor = None,
            requires_safety_checker = False,
        )
        images = pipe_inpainting(
            prompt = prompt,
            image = initimage,
            mask_image = imagemask,
            width = width,
            height = height,
            strength = strength,
            num_inference_steps = steps,
            guidance_scale = cfg,
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
            cross_attention_kwargs = None)

    img = images[0][0]
    del images
    del initimage
    del imagemask
    del pipe_img2img
    del pipe_inpainting
    del generator
    log.info("image generated successfully.")
    return img


def create_image(
    scheduler,
    prompt, negative,
    width, height,
    seed, steps, cfg, strength,
    lora,
    initimage, imagemask,
    facefix, upscale,
    savefile, onefile, outpath, filename):

    set_scheduler(scheduler)
    
    if initimage:
        if initimage.startswith('data:image/png'): # from base64
            initimage = utils.base64Decode(initimage)
            log.info("initial image [base64]")
        elif initimage.endswith('.png') or initimage.endswith('.PNG'): # file path
            log.info(f"initial image [{ initimage }]")
            initimage = Image.open(initimage)
        else:
            log.error(f"bad initial image [{ initimage[:50] }]") # bad input
            return None
        initimage = initimage.resize((width, height))
        initimage = initimage.convert("RGB")

    if imagemask:
        if imagemask.startswith('data:image/png'): # from base64
            imagemask = utils.base64Decode(imagemask)
            log.info("image mask [base64]")
        elif imagemask.endswith('png') or imagemask.endswith('PNG'): # file path
            log.info(f"image mask [{ imagemask }]")
            imagemask = Image.open(imagemask)
        else:
            log.error(f"bad image mask [{ imagemask[:50] }]") # bad input
            return None
        imagemask = imagemask.resize((width, height))
        imagemask = imagemask.convert("RGB")
            
    img = generate_image(
        prompt, negative,
        int(width), int(height),
        int(seed), int(steps), float(cfg), float(strength),
        float(lora),
        initimage, imagemask)

    del initimage
    del imagemask

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
        "cfg": cfg,
        "strength": strength,
        "lora": lora,
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
        log.info("apply filter [gfpgan]...")
        img = inference_gfpgan(img, width=width, height=height, upscale=1)
        if savefile and not onefile:
            img.save(f"{ savepath }_ff.png", pnginfo=metadata)
            log.info(f"image saved to { savepath }_ff.png")

    if upscale:
        log.info("apply filter [realesrgan 4x]...")
        img = inference_realesrgan(img, width=width, height=height, scale=4)
        if savefile and not onefile:
            img.save(f"{ savepath }_up.png", pnginfo=metadata)
            log.info(f"image saved to { savepath }_up.png")

    if savefile and onefile:
        img.save(f"{ savepath }.png", pnginfo=metadata)
        log.info(f"image saved to { savepath }.png")

    out_base64 = utils.base64Encode(img, "png")

    del img
    del metadata
    log.info("done")
    return [ metadict, out_base64 ]


from gc import collect
def clear_cache():
    collect()
    if cuda.is_available():
        cuda.empty_cache()
        cuda.ipc_collect()

def clear_pipe():
    if ctx.pipe:
        del ctx.pipe
    clear_cache()
