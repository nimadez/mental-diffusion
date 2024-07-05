#!/usr/bin/env python3
# https://github.com/nimadez/mental-diffusion/
#
# pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
# pip install realesrgan
# python3 src/addons/upscale.py
#
# Auto-download models to ~/.cache/realesrgan (64MB each model)


USER = __import__('getpass').getuser()
MODEL_CACHE = f"/home/{USER}/.cache/realesrgan"
URLS = [
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
]


# torchvision 0.17+ basicsr workaround
__import__('warnings').filterwarnings("ignore", category=UserWarning) # disable esrgan/torchvision 0.16 warnings
import sys
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as functional
        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass


import torch, gc, os, numpy
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


def download_realesrgan(url, model_path):
    if not os.path.exists(model_path):
        print(f"Downloading {os.path.basename(model_path)} ...")
        if not os.path.exists(MODEL_CACHE):
            os.mkdir(MODEL_CACHE)
        torch.hub.download_url_to_file(url, model_path, progress=True)


def inference_realesrgan(model_type, img_path, output):
    model = None
    model_name = None
    model_path = None
    netscale = 0
    match model_type:
        case "x2":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            model_name = "RealESRGAN_x2plus.pth"
            model_path = f"{MODEL_CACHE}/{model_name}"
            netscale = 2
            download_realesrgan(URLS[0], model_path)
        case "x4":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_name = "RealESRGAN_x4plus.pth"
            model_path = f"{MODEL_CACHE}/{model_name}"
            netscale = 4
            download_realesrgan(URLS[1], model_path)

    if not os.path.exists(model_path):
        print("ERROR: Model does not exists, download failed.")
        sys.exit()

    print("Upscaler:", model_name)
    upsampler = RealESRGANer(
        scale = netscale,
        model_path = model_path,
        dni_weight = None,
        model = model,
        tile = 256, # ~512
        tile_pad = 10,
        pre_pad = 0,
        half = "fp16",
        gpu_id = None) # 0,1,2

    image = Image.open(img_path).convert("RGB")
    image = numpy.array(image, dtype=numpy.uint8)[..., ::-1]
    img, _ = upsampler.enhance(image, outscale=netscale)
    img = img[:, :, ::-1]
    img = Image.fromarray(img)

    savepath = f"{output}/{os.path.basename(img_path)}_x{netscale}.png"
    img.save(savepath)
    print(f"Saved to {savepath}")


if __name__== "__main__":
    if len(sys.argv) > 3:
        model = sys.argv[1]
        image = sys.argv[2]
        output = sys.argv[3]

        if model not in ['x2', 'x4']:
            print("ERROR: Model type does not exists. [ x2, x4 ]")
            sys.exit()

        if os.path.exists(output):
            if os.path.exists(image):
                inference_realesrgan(model, image, output)
            else:
                print("ERROR: Image does not exist.")
        else:
            print("ERROR: Output path does not exist.")
    else:
        print('help: python3 upscale.py [x2/x4] [image-path] [output-directory]')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
