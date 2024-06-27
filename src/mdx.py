#!/usr/bin/env python3
VER = "0.8.6" # https://github.com/nimadez/mental-diffusion


USER = __import__('getpass').getuser()
DEBUG = False
PROXY = None
OFFLINE = False
USE_CPU = False
OUTPUT = f"/home/{USER}/.mdx"
SAVE_ANIM = False
MODEL_TYPE = "xl"
CHECKPOINT = f"/media/{USER}/storage/sd/model_sdxl.safetensors"
WIDTH, HEIGHT = 768, 768
SCHEDULER = "euler"


import os
if PROXY:
    os.environ["http_proxy"] = os.environ["https_proxy"] = PROXY
if OFFLINE:
    os.environ["DISABLE_TELEMETRY"] = "YES"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
if not DEBUG:
    os.environ["DIFFUSERS_VERBOSITY"] = "error"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
    __import__('warnings').filterwarnings("ignore", category=FutureWarning)
    __import__('transformers').logging.set_verbosity_error()
__import__('colorama').init(autoreset=True)


import torch, gc, sys, json, math, logging, random, time
from colorama import Fore
from argparse import ArgumentParser
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers import (AutoencoderKL, AutoencoderTiny,
    StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline,
    DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, LCMScheduler, LMSDiscreteScheduler, PNDMScheduler)


def arg_parser(args):
    parser = ArgumentParser("mdx.py", add_help = False)
    parser.add_argument('--help', action = "help", help = "show this help message and exit")
    parser.add_argument('-t',  '--type', type = str, default = MODEL_TYPE, required = False, help = f"sd, xl, checkpoint model type (def: {MODEL_TYPE})")
    parser.add_argument('-c',  '--checkpoint', type = str, default = CHECKPOINT, required = False, help = f"checkpoint.safetensors (def: {os.path.basename(CHECKPOINT)})")
    parser.add_argument('-sc', '--scheduler', type = str, default = SCHEDULER, help = f"ddim, ddpm, lcm, pndm, eulera, euler, lms (def: {SCHEDULER})")
    parser.add_argument('-p',  '--prompt', type = str, default = "", help = "positive prompt text input (def: empty)")
    parser.add_argument('-n',  '--negative', type = str, default = "", help = "negative prompt text input (def: empty)")
    parser.add_argument('-w',  '--width', type = int, default = WIDTH, help = f"width value must be divisible by 8 (def: {WIDTH})")
    parser.add_argument('-h',  '--height', type = int, default = HEIGHT, help = f"height value must be divisible by 8 (def: {HEIGHT})")
    parser.add_argument('-s',  '--seed', type = int, default = -1, help = "seed number, -1 randomize (def: -1)")
    parser.add_argument('-st', '--steps', type = int, default = 24, help = "steps from 1 to 100+ (def: 24)")
    parser.add_argument('-g',  '--guidance', type = float, default = 8.0, help = "0-20.0+, how closely linked to the prompt (def: 8.0)")
    parser.add_argument('-sr', '--strength', type = float, default = 1.0, help = "0-1.0, how much respect the image should pay to the original (def: 1.0)")
    parser.add_argument('-ls', '--lorascale', type = float, default = 1.0, help = "0-1.0, lora scale (def: 1.0)")
    parser.add_argument('-i',  '--image', type = str, default = None, help = "image.png (def: none)")
    parser.add_argument('-m',  '--mask', type = str, default = None, help = "mask.png (def: none)")
    parser.add_argument('-v',  '--vae', type = str, default = None, required = False, help = "vae.safetensors (def: none)")
    parser.add_argument('-l',  '--lora', type = str, default = None, help = "lora.safetensors (def: none)")
    parser.add_argument('-f',  '--filename', type = str, default = "img_{seed}", help = "filename prefix without .png extension, add {seed} to be replaced (def: img_{seed})")
    parser.add_argument('-no', '--number', type = int, default = 1, help = "number of images to generate per prompt (def: 1)")
    parser.add_argument('-b',  '--batch', type = int, default = 0, help = "number of repeats to run in batch, --seed -1 to randomize")
    parser.add_argument('-pv', '--preview', action = 'store_true', help = "stepping is slower with preview enabled (def: no preview)")
    parser.add_argument('-lv', '--lowvram', action = 'store_true', help = "slower if you have enough VRAM, automatic on 4GB cards (def: no lowvram)")
    parser.add_argument('-meta', '--metadata', type = str, default = None, help = "image.png, extract metadata from png")    
    return parser.parse_args(args)


class MDX():
    def __init__(self):
        self.md = []
        self.tae = None
        self.frames = []
        self.encoded_prompt = []

        self.dtype = torch.float16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if USE_CPU or self.device == "cpu":
            self.dtype = None
            self.device = "cpu"


    def auto_pipeline(self, variant, pipe=None):
        if not self.md.image and not self.md.mask:
            if self.md.type == "sd": pipe = StableDiffusionPipeline.from_single_file(self.md.checkpoint, torch_dtype=self.dtype, variant=variant, local_files_only=False, use_safetensors=True)
            elif self.md.type == "xl": pipe = StableDiffusionXLPipeline.from_single_file(self.md.checkpoint, torch_dtype=self.dtype, variant=variant, local_files_only=False, use_safetensors=True)
            self.md.pipeline = "txt2img"
        elif self.md.image and not self.md.mask:
            if self.md.type == "sd": pipe = StableDiffusionImg2ImgPipeline.from_single_file(self.md.checkpoint, torch_dtype=self.dtype, variant=variant, local_files_only=False, use_safetensors=True)
            elif self.md.type == "xl": pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(self.md.checkpoint, torch_dtype=self.dtype, variant=variant, local_files_only=False, use_safetensors=True)
            self.md.pipeline = "img2img"
        elif self.md.image and self.md.mask:
            if self.md.type == "sd": pipe = StableDiffusionInpaintPipeline.from_single_file(self.md.checkpoint, torch_dtype=self.dtype, variant=variant, local_files_only=False, use_safetensors=True)
            elif self.md.type == "xl": pipe = StableDiffusionXLInpaintPipeline.from_single_file(self.md.checkpoint, torch_dtype=self.dtype, variant=variant, local_files_only=False, use_safetensors=True)
            self.md.pipeline = "inpaint"
        return pipe


    def load_pipeline(self, variant="fp16"):
        if self.md.preview:
            if self.md.type == "sd": self.tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=self.dtype)
            if self.md.type == "xl": self.tae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=self.dtype)
        
        print("Loading checkpoint:", Fore.CYAN + os.path.basename(self.md.checkpoint))
        pipe = self.auto_pipeline(variant)
        if pipe:
            if self.md.vae:
                print("Loading vae:", Fore.CYAN + os.path.basename(self.md.vae))
                pipe.vae = AutoencoderKL.from_single_file(self.md.vae, torch_dtype=self.dtype, variant=variant, local_files_only=False, use_safetensors=True).to(self.device, self.dtype)
                
            if self.md.lora:
                print("Loading lora:", Fore.CYAN + os.path.basename(self.md.lora))
                pipe.load_lora_weights(os.path.dirname(self.md.lora), weight_name=os.path.basename(self.md.lora), adapter_name="default")

            pipe.enable_sequential_cpu_offload() if self.device != "cpu" and self.md.lowvram else pipe.to(self.device, self.dtype)
            pipe.enable_attention_slicing(1)
            if self.device != "cpu":
                pipe.enable_vae_slicing()
                pipe.enable_vae_tiling()
                if self.tae:
                    self.tae = self.tae.half().eval().requires_grad_(False).to(self.device, self.dtype)
                    self.tae.enable_slicing()
                    self.tae.enable_tiling()
        return pipe


    def load_scheduler(self, config):
        match self.md.scheduler:
            case "ddim": return DDIMScheduler.from_config(config)
            case "ddpm": return DDPMScheduler.from_config(config)
            case "euler": return EulerDiscreteScheduler.from_config(config)
            case "eulera": return EulerAncestralDiscreteScheduler.from_config(config)
            case "lcm": return LCMScheduler.from_config(config)            
            case "lms": return LMSDiscreteScheduler.from_config(config, use_karras_sigmas=True)
            case "pndm": return PNDMScheduler.from_config(config)


    def callback_on_step_end(self, pipe, idx, ts, callback_kwargs):
        latents = callback_kwargs["latents"]
        with torch.no_grad():
            latent_model_input = pipe.scheduler.scale_model_input(latents, ts)
            noise_pred = pipe.unet(latent_model_input, ts, encoder_hidden_states=self.encoded_prompt[0]).sample
            
            if not hasattr(pipe.scheduler, 'sigmas'): # non-discretes
                alpha_t = torch.sqrt(pipe.scheduler.alphas_cumprod)[ts]
                sigma_t = torch.sqrt(1 - pipe.scheduler.alphas_cumprod)[ts]
                latents = (latents - sigma_t * noise_pred) / alpha_t
            else: # discretes
                step_index = (pipe.scheduler.timesteps == ts).nonzero().item()
                sigma_t = pipe.scheduler.sigmas[step_index + 1]
                latents = latents - sigma_t * noise_pred

            latents = 1 / self.tae.config.scaling_factor * latents
            decoded = self.tae.decode(latents).sample
            image = pipe.image_processor.postprocess(decoded)[0]
            image.save(OUTPUT + "/preview.jpg", format="jpeg", quality=100, optimize=False, progressive=False)
            self.frames.append(image)
        return callback_kwargs


    def inference(self, pipe):
        width, height = round(self.md.width / 8) * 8, round(self.md.height / 8) * 8
        seed, steps, image, mask = self.md.seed, self.md.steps, self.md.image, self.md.mask
        if seed == -1:
            seed = random.randrange(0, sys.maxsize)
        if steps * self.md.strength < 1:
            steps = math.ceil(1 / max(0.1, self.md.strength))
        if image:
            image = Image.open(image).convert("RGB")
            width, height = image.size
            print("Load image:", Fore.CYAN + self.md.image)
        if mask:
            mask = Image.open(mask).convert("RGB")
            print("Load mask:", Fore.CYAN + self.md.mask)

        if pipe:
            pipe.scheduler = self.load_scheduler(pipe.scheduler.config)
            pipe.scheduler.set_timesteps(steps)
            generator = torch.Generator("cpu").manual_seed(seed)

            if self.md.preview:
                if self.md.type == "xl": pipe.unet.config.addition_embed_type = None
                self.encoded_prompt = pipe.encode_prompt(device=self.device, prompt=self.md.prompt, num_images_per_prompt=self.md.number, do_classifier_free_guidance=self.md.guidance > 1.0)
                self.frames = []

            print(f"[{self.md.type}, {self.md.pipeline}, {width}x{height}, {self.md.scheduler}, {steps}, {self.md.guidance}, {self.md.strength}, {self.md.lorascale}, +{self.md.number-1}, {seed}]")
            images = pipe(output_type="pil",
                prompt = self.md.prompt,
                negative_prompt = self.md.negative,
                width = width,
                height = height,
                generator = generator,
                num_inference_steps = steps,
                guidance_scale = self.md.guidance,
                strength = self.md.strength,
                lora_scale = self.md.lorascale,
                image = image,
                mask_image = mask,
                num_images_per_prompt = self.md.number,
                cross_attention_kwargs = { "scale": self.md.lorascale } if self.md.lora else None,
                callback_on_step_end = self.callback_on_step_end if self.md.preview else None,
                callback_on_step_end_tensor_inputs = ['latents'])[0]

            pngInfo = PngInfo()
            pngInfo.add_text("MDX", json.dumps({ "version": VER, "type": self.md.type, "pipeline": self.md.pipeline,
                "checkpoint": os.path.abspath(self.md.checkpoint), "vae": os.path.abspath(self.md.vae) if self.md.vae else None, "lora": os.path.abspath(self.md.lora) if self.md.lora else None,
                "scheduler": self.md.scheduler, "seed": seed, "steps": steps, "guidance": self.md.guidance, "strength": self.md.strength, "lorascale": self.md.lorascale,
                "prompt": self.md.prompt, "negative": self.md.negative, "image": os.path.abspath(self.md.image) if self.md.image else None, "mask": os.path.abspath(self.md.mask) if self.md.mask else None
            }))

            savepath = None
            for img in images:
                count = 1
                filename = self.md.filename.replace("{seed}", str(seed))
                savepath = f"{OUTPUT}/{filename}"
                while os.path.exists(f"{savepath}.png"):
                    savepath = f"{OUTPUT}/{filename}_{str(count)}"
                    count += 1
                img.save(f"{savepath}.png", pnginfo=pngInfo)
                print("Saved:", Fore.GREEN + f"{savepath}.png")

            if SAVE_ANIM and self.md.preview:
                self.frames.append(img)
                self.frames[0].save(f"{savepath}.webp", append_images=self.frames[1:], save_all=True, lossless=False, quality=100, method=5, duration=300)
                print("Saved:", Fore.GREEN + f"{savepath}.webp")
        else:
            print(Fore.RED + "ERROR: Failed to load pipeline.")


    def main(self, args):
        self.md = args
        if self.device != "cpu" and self.md.type != "sd" and torch.cuda.get_device_properties(self.device).total_memory < 5000000000: self.md.lowvram = True
        print(Fore.MAGENTA + f"MDX {VER}", self.device.upper(), "LV", "PV" if self.md.preview else "") if self.device != "cpu" and self.md.lowvram else print(Fore.MAGENTA + f"MDX {VER}", self.device.upper(), "PV" if self.md.preview else "")
        self.clear_cache()

        try:
            pipe = self.load_pipeline()
            if self.md.batch > 0:
                for _ in range(1, self.md.batch + 1):
                    print(Fore.CYAN + f"#{_}")
                    self.inference(pipe)
                    self.clear_cache()
                    time.sleep(1)
            else:
                self.inference(pipe)
        except KeyboardInterrupt:
            print("(0)")

        self.clear_cache()


    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()


if __name__== "__main__":
    if len(sys.argv) > 1:
        args = arg_parser(sys.argv[1:])
        
        if args.metadata != None:
            if os.path.exists(args.metadata):
                print(Image.open(args.metadata).info); sys.exit(0)
            print(Fore.RED + "ERROR: --metadata image does not exists."); sys.exit(1)

        if not os.path.exists(OUTPUT): os.mkdir(OUTPUT)
        if args.checkpoint and not os.path.exists(args.checkpoint):
            print(Fore.RED + "ERROR: --checkpoint does not exists."); sys.exit(1)
        if args.image and not os.path.exists(args.image):
            print(Fore.RED + "ERROR: --image does not exists."); sys.exit(1)
        if args.mask and not os.path.exists(args.mask):
            print(Fore.RED + "ERROR: --mask image does not exists."); sys.exit(1)
        if args.vae and not os.path.exists(args.vae):
            print(Fore.RED + "ERROR: --vae model does not exists."); sys.exit(1)
        if args.lora and not os.path.exists(args.lora):
            print(Fore.RED + "ERROR: --lora model does not exists."); sys.exit(1)
        if args.scheduler not in ["ddim", "ddpm", "euler", "eulera", "lcm", "lms", "pndm"]:
            print(Fore.RED + "ERROR: --scheduler does not exists."); sys.exit(1)

        MDX().main(args)
    else:
        print("help: python3 mdx.py --help")
