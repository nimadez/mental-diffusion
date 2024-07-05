#
# a prototype for SD3 implantation
#
# install new python packages:
# $ pip install sentencepiece protobuf
#
# supported checkpoint: sd3_medium_incl_clips_t5xxlfp8
# https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium_incl_clips_t5xxlfp8.safetensors


from diffusers import (AutoencoderKL, AutoencoderTiny,
    #...
    StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline,
    #...)


def arg_parser(args):
    #...
    parser.add_argument('-t',  '--type', type = str, default = MODEL_TYPE, required = False, help = f"sd, xl, 3, checkpoint model type (def: {MODEL_TYPE})")
    #...


def auto_pipeline(self, variant, pipe=None):
    if not self.md.image and not self.md.mask:
        #...
        elif self.md.type == "3": pipe = StableDiffusion3Pipeline.from_single_file(self.md.checkpoint, torch_dtype=self.dtype, variant=variant, local_files_only=False, use_safetensors=True)
        #...
    elif self.md.image and not self.md.mask:
        #...
        elif self.md.type == "3": pipe = StableDiffusion3Img2ImgPipeline.from_single_file(self.md.checkpoint, torch_dtype=self.dtype, variant=variant, local_files_only=False, use_safetensors=True)
        #...
    elif self.md.image and self.md.mask:
        #...
        elif self.md.type == "3": return None
        #...


def load_pipeline(self, variant="fp16"): # TODO: not sure about the variant
    if self.md.preview:
        #if self.md.type == "sd": self.tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=self.dtype)
        else: self.tae = AutoencoderTiny.from_pretrained("madebyollin/taesd" + self.md.type, torch_dtype=self.dtype)
    #...
    #pipe.enable_attention_slicing(1)
    if self.device != "cpu" and self.md.type != "3": # no vae_slicing for SD3
    #    pipe.enable_vae_slicing()
    #    pipe.enable_vae_tiling()


def inference(self, pipe):
    if pipe:
        #...
        if self.md.preview:
            #if self.md.type == "xl": pipe.unet.config.addition_embed_type = None
            if self.md.type == "3":  pipe.vae.config.shift_factor = 0.0
            #...
        #...
        images = pipe(
            prompt = self.md.prompt,
            negative_prompt = self.md.negative,
            width = width,
            height = height,
            generator = generator,
            num_inference_steps = steps,
            guidance_scale = self.md.guidance,
            #strength = self.md.strength,       # <-- remove all commented lines
            #lora_scale = self.md.lorascale,    # None **kwargs for SD3 pipelines are
            #image = image,                     # not implanted in diffusers yet
            #mask_image = mask,
            num_images_per_prompt = self.md.number,
            #cross_attention_kwargs = { "scale": self.md.lorascale } if self.md.lora else None,
            callback_on_step_end = self.callback_on_step_end if self.md.preview else None,
            callback_on_step_end_tensor_inputs = ['latents'])[0]

# TODO: not sure if the output is fine with fp16 variant
