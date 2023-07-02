#
# mental-diffusion utils
#


# ----- base64 PNG encoder/decoder -----


from base64 import b64encode, b64decode
from io import BytesIO
from PIL import Image

def base64Encode(img, format="png"):
    buffered = BytesIO()
    img.save(buffered, format=format)
    buffered.seek(0)
    b64str = f"data:image/{ format };base64," + b64encode(buffered.getvalue()).decode()
    del buffered
    return b64str

def base64Decode(str):
    return Image.open(BytesIO(b64decode(str.split(',')[1])))


# ----- get memory stats -----


def mem_stats_total():
    from psutil import cpu_percent, virtual_memory
    from torch.cuda import mem_get_info, is_available
    ram_total = virtual_memory().total
    vram_total = mem_get_info("cuda:0")[0] if is_available() else 0
    ram_total /= 1024 ** 3
    vram_total /= 1024 ** 3
    return [ ram_total, vram_total ]


# ----- check huggingface cache -----


def hf_cache_check(model, filename):
    from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
    cache = try_to_load_from_cache(model, filename=filename)
    if isinstance(cache, str):
        return True
    elif cache is _CACHED_NO_EXIST:
        return False
    else:
        return False


# ----- custom checkpoint loader for safetensors -----
# https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/loaders.py
# add "original_config_file" to be able to load the checkpoint 100% offline
# https://github.com/huggingface/diffusers/issues/3729


def pipe_from_ckpt(cls, pretrained_model_link_or_path, **kwargs):
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
    from diffusers.utils import (
        DIFFUSERS_CACHE,
        HF_HUB_OFFLINE,
        TEXT_ENCODER_ATTN_MODULE,
        is_safetensors_available
    )

    original_config_file = kwargs.pop("original_config_file", None) # * NEW ADDED LINE *
    cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
    resume_download = kwargs.pop("resume_download", False)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    extract_ema = kwargs.pop("extract_ema", False)
    image_size = kwargs.pop("image_size", 512)
    scheduler_type = kwargs.pop("scheduler_type", "pndm")
    num_in_channels = kwargs.pop("num_in_channels", None)
    upcast_attention = kwargs.pop("upcast_attention", None)
    load_safety_checker = kwargs.pop("load_safety_checker", True)
    prediction_type = kwargs.pop("prediction_type", None)
    text_encoder = kwargs.pop("text_encoder", None)
    tokenizer = kwargs.pop("tokenizer", None)

    torch_dtype = kwargs.pop("torch_dtype", None)

    use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

    pipeline_name = cls.__name__
    file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
    from_safetensors = file_extension == "safetensors"

    if from_safetensors and use_safetensors is False:
        raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

    # TODO: For now we only support stable diffusion
    stable_unclip = None
    model_type = None
    controlnet = False

    if pipeline_name == "StableDiffusionControlNetPipeline":
        # Model type will be inferred from the checkpoint.
        controlnet = True
    elif "StableDiffusion" in pipeline_name:
        # Model type will be inferred from the checkpoint.
        pass
    elif pipeline_name == "StableUnCLIPPipeline":
        model_type = "FrozenOpenCLIPEmbedder"
        stable_unclip = "txt2img"
    elif pipeline_name == "StableUnCLIPImg2ImgPipeline":
        model_type = "FrozenOpenCLIPEmbedder"
        stable_unclip = "img2img"
    elif pipeline_name == "PaintByExamplePipeline":
        model_type = "PaintByExample"
    elif pipeline_name == "LDMTextToImagePipeline":
        model_type = "LDMTextToImage"
    else:
        raise ValueError(f"Unhandled pipeline class: {pipeline_name}")

    # remove huggingface url
    for prefix in ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]:
        if pretrained_model_link_or_path.startswith(prefix):
            pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]

    # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
    ckpt_path = Path(pretrained_model_link_or_path)
    if not ckpt_path.is_file():
        # get repo_id and (potentially nested) file path of ckpt in repo
        repo_id = "/".join(ckpt_path.parts[:2])
        file_path = "/".join(ckpt_path.parts[2:])

        if file_path.startswith("blob/"):
            file_path = file_path[len("blob/") :]

        if file_path.startswith("main/"):
            file_path = file_path[len("main/") :]

        pretrained_model_link_or_path = hf_hub_download(
            repo_id,
            filename=file_path,
            cache_dir=cache_dir,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
        )

    pipe = download_from_original_stable_diffusion_ckpt(
        pretrained_model_link_or_path,
        original_config_file=original_config_file, # * NEW ADDED LINE *
        pipeline_class=cls,
        model_type=model_type,
        stable_unclip=stable_unclip,
        controlnet=controlnet,
        from_safetensors=from_safetensors,
        extract_ema=extract_ema,
        image_size=image_size,
        scheduler_type=scheduler_type,
        num_in_channels=num_in_channels,
        upcast_attention=upcast_attention,
        load_safety_checker=load_safety_checker,
        prediction_type=prediction_type,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    if torch_dtype is not None:
        pipe.to(torch_dtype=torch_dtype)

    return pipe


# ----- load VAE weight from safetensors -----


def load_vae_weights(vae_path, device, samplesize, modelconfig):
    from io import BytesIO
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    from diffusers import AutoencoderKL
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        create_vae_diffusers_config,
        convert_ldm_vae_checkpoint
    )

    # convert the VAE model
    state_dict = load_file(vae_path, device=device)
    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith("first_stage_model."):
            key = "first_stage_model." + key
        new_state_dict[key] = value

    # load config
    with open(modelconfig, "rb") as f:
        buf = BytesIO(f.read())
    original_config = OmegaConf.load(buf)

    vae_config = create_vae_diffusers_config(original_config, image_size=samplesize) # def: 512
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(new_state_dict, vae_config)

    # encode vae
    if len(converted_vae_checkpoint) > 0:
        vae = AutoencoderKL.from_config(vae_config)
        vae.load_state_dict(converted_vae_checkpoint)
        return vae
    return None
