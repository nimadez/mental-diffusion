#!/usr/bin/env python3
# Aug 2024 | Mental Diffusion | https://github.com/nimadez/mental-diffusion
#
# Real-ESRGAN upscaler x2 and x4 plus

# Inference:
# python3 mdx-upscale.py --help
# python3 mdx-upscale.py -i ./image.png
# python3 mdx-upscale.py -i ./image.png -m x2
# python3 mdx-upscale.py -i ./image.png -m x4 -o ~/Downloads
#
# - Auto-download models (64 MB each)


USER = __import__('getpass').getuser()
MODEL_CACHE = f"/home/{USER}/.cache/realesrgan"
URLS = [ "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
         "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" ]


import os, sys, gc, torch, numpy
from PIL import Image
from argparse import ArgumentParser

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/libs')
from realesrgan import RealESRGANer, RRDBNet


def arg_parser(args):
    parser = ArgumentParser("mdx-upscale.py", add_help = False)
    parser.add_argument('--help', action = "help", help = "show this help message and exit")
    parser.add_argument('-m', '--model', type = str, default = "x2", help = "x2 or x4 (def: x2)")
    parser.add_argument('-i', '--image', type = str, default = None, help = "image path (def: None)")
    parser.add_argument('-o', '--output', type = str, default = ".", help = "output directory (def: .)")
    return parser.parse_args(args)


class Upscaler():
    def __init__(self):
        pass


    def download_realesrgan(self, url, model_path):
        if not os.path.exists(model_path):
            print(f"Downloading {os.path.basename(model_path)} ...")
            if not os.path.exists(MODEL_CACHE):
                os.mkdir(MODEL_CACHE)
            torch.hub.download_url_to_file(url, model_path, progress=True)

            
    def inference(self, model_type, img_path, output):
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
                self.download_realesrgan(URLS[0], model_path)
            case "x4":
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                model_name = "RealESRGAN_x4plus.pth"
                model_path = f"{MODEL_CACHE}/{model_name}"
                netscale = 4
                self.download_realesrgan(URLS[1], model_path)

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

        if os.path.exists(output):
            fpath = f"{output}/{os.path.basename(img_path)}_x{netscale}.png"
            img.save(fpath)
            print("Saved:", fpath)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()


if __name__== "__main__":
    if len(sys.argv) > 1:
        args = arg_parser(sys.argv[1:])

        if not os.path.exists(args.output):
            print("ERROR: Output directory does not exist.")
            sys.exit()

        if args.model in ["x2", "x4"]:
            if os.path.exists(args.image):
                Upscaler().inference(args.model, args.image, args.output)
            else:
                print("ERROR: Image does not exist.")
        else:
            print("ERROR: Invalid model, use x2 or x4.")
    else:
        print("help: python3 mdx-upscale.py --help")
