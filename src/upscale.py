#!/usr/bin/env python3
#
# Real-ESRGAN upscaler x4 plus
#
# $ pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
# $ pip install realesrgan
#
# Automatic download to ~/.cache/realesrgan (67MB)


URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL = f"/home/{__import__('getpass').getuser()}/.cache/realesrgan/RealESRGAN_x4plus.pth"


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


def upscale(img_path):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    print("Upscaler:", os.path.basename(MODEL))

    upsampler = RealESRGANer(
        scale = netscale,
        model_path = MODEL,
        dni_weight = None,
        model = model,
        tile = 256, # ~512
        tile_pad = 10,
        pre_pad = 0,
        half = "fp16",
        gpu_id = None) # 0,1,2

    image = Image.open(img_path)
    image = image.convert("RGB")
    image = numpy.array(image, dtype=numpy.uint8)[..., ::-1]
    img, _ = upsampler.enhance(image, outscale=netscale)
    img = img[:, :, ::-1]
    img = Image.fromarray(img)

    savepath = f"{os.path.basename(img_path)}_x4.png"
    img.save(savepath)
    print(f"Saved to {savepath}")


if __name__== "__main__":
    if not os.path.exists(MODEL):
        print(f"Downloading {os.path.basename(MODEL)} ...")
        os.mkdir(os.path.dirname(MODEL))
        torch.hub.download_url_to_file(URL, MODEL, progress=True)

    if len(sys.argv) > 1:
        image = sys.argv[1]
        if os.path.exists(image):
            upscale(image)
        else:
            print("ERROR: Image does not exist.")
    else:
        print("help: python3 upscale.py [./image-path]")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
