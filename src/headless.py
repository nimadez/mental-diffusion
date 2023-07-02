#
# mental-diffusion headless
#
import sys
import json
import random
from argparse import ArgumentParser
from websockets.sync.client import connect
from PIL import Image


MAX_SIZE_BYTES = 2 ** 25 # 33MB


def ws_connect(key, val, out=None):
    with connect("ws://localhost:8011/index", max_size=MAX_SIZE_BYTES) as ws:
        ws.send(json.dumps({
            "key": key,
            "val": val
        }))
        try:
            if out == None:
                print(ws.recv(1024))
            elif out == "metadata":
                print(json.dumps( json.loads(ws.recv(1024))["metadata"] ))
            elif out == "base64":
                print(json.dumps( json.loads(ws.recv(1024))["base64"] ))
        except:
            print("cancelled.")
            sys.exit(0)


def main(args):
    parser = ArgumentParser('headless.py', add_help=False)
    parser.add_argument(
        '--help',
        action='help',
        help='show this help message and exit')

    parser.add_argument(
        '-g', '--get',
        type = int,
        default = 0,
        help = "fetch data from server by record id (int)"
    )

    parser.add_argument(
        '-b', '--batch',
        type = int,
        default = 1,
        help = "number of images to render (def: 1)"
    )
    parser.add_argument(
        '-sc', '--scheduler',
        type = str,
        default = "euler_anc",
        help = "ddpm, ddim, pndm, lms, euler, euler_anc (def: euler_anc)"
    )
    parser.add_argument(
        '-p', '--prompt',
        type = str,
        default = "a photo of an astronaut riding a horse",
        help = "positive prompt input"
    )
    parser.add_argument(
        '-n', '--negative',
        type = str,
        default = "deformed, distorted, disfigured, poorly drawn, poorly drawn hands, poorly drawn face, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, text, label, word, malformed hands, out of focus",
        help = "negative prompt input"
    )
    parser.add_argument(
        '-w', '--width',
        type = int,
        default = 512,
        help = "image width have to be divisible by 8 (def: 512)"
    )
    parser.add_argument(
        '-h', '--height',
        type = int,
        default = 512,
        help = "image height have to be divisible by 8 (def: 512)"
    )
    parser.add_argument(
        '-s', '--seed',
        type = int,
        default = 0,
        help = "seed number, 0 to randomize (def: 0)"
    )
    parser.add_argument(
        '-st', '--steps',
        type = int,
        default = 20,
        help = "steps from 10 to 50, 20-25 is good enough (def: 20)"
    )
    parser.add_argument(
        '-c', '--cfg',
        type = float,
        default = 7.5,
        help = "guidance scale, how closely linked to the prompt (def: 7.5)"
    )
    parser.add_argument(
        '-sr', '--strength',
        type = float,
        default = 0.5,
        help = "how much respect the final image should pay to the original (def: 0.5)"
    )
    parser.add_argument(
        '-l', '--lora',
        type = float,
        default = 0.0,
        help = "TODO (def: 0.0)"
    )
    parser.add_argument(
        '-i', '--image',
        type = str,
        default = "",
        help = "PNG file path or base64 PNG (def: '')"
    )
    parser.add_argument(
        '-m', '--mask',
        type = str,
        default = "",
        help = "PNG file path or base64 PNG (def: '')"
    )
    parser.add_argument(
        '-ff', '--facefix',
        type = lambda x: (str(x).lower() == 'true'),
        default = False,
        help = "true/false, face restoration using gfpgan (def: false)"
    )
    parser.add_argument(
        '-up', '--upscale',
        type = lambda x: (str(x).lower() == 'true'),
        default = False,
        help = "true/false, upscale using real-esrgan 4x (def: false)"
    )
    parser.add_argument(
        '-sv', '--savefile',
        type = lambda x: (str(x).lower() == 'true'),
        default = True,
        help = "true/false, save image to PNG, contain metadata (def: true)"
    )
    parser.add_argument(
        '-of', '--onefile',
        type = lambda x: (str(x).lower() == 'true'),
        default = False,
        help = "true/false, save the final result only (def: false)"
    )
    parser.add_argument(
        '-o', '--outpath',
        type = str,
        default = "./.output",
        help = "/path-to-directory (def: ./.output)"
    )
    parser.add_argument(
        '-f', '--filename',
        type = str,
        default = "img",
        help = "filename prefix (.png extension is not required)"
    )

    parser.add_argument(
        '-ckpt', '--ckpt',
        type = str,
        default = "null",
        help = "change/reload checkpoint by name"
    )
    parser.add_argument(
        '-meta', '--metadata',
        type = str,
        default = "null",
        help = "/path-to-image.png, extract metadata from PNG"
    )
    parser.add_argument(
        '-out', '--out',
        type = str,
        default = "metadata",
        help = "stdout 'metadata' or 'base64' (def: metadata)"
    )

    args = parser.parse_args(args)

    if args.get != 0:
        ws_connect("GET", args.get)
        return

    if args.ckpt != "null":
        ws_connect("MOD", args.ckpt)
        return

    if args.metadata != "null":
        get_metadata(args.metadata)
        return

    # width and height have to be divisible by 8
    args.width = round(args.width / 8) * 8
    args.height = round(args.height / 8) * 8

    if args.image != "" and args.mask != "":
        if args.steps < 10 and args.strength < 0.2:
            print("Error: the input is incorrect for inpainting pipeline")
            return

    if args.seed == 0:
        args.seed = random.randrange(0, 4294967295)

    filename = f"{ args.filename }_{ args.seed }_{ args.steps }_{ args.cfg }"

    # create image
    data = {
        'scheduler':    args.scheduler,
        'prompt':       args.prompt,
        'negative':     args.negative,
        'width':        args.width,
        'height':       args.height,
        'seed':         args.seed,
        'steps':        args.steps,
        'cfg':          args.cfg,
        'strength':     args.strength,
        'lora':         args.lora,
        'image':        args.image,
        'mask':         args.mask,
        'facefix':      args.facefix,
        'upscale':      args.upscale,
        'savefile':     args.savefile,
        'onefile':      args.onefile,
        'outpath':      args.outpath,
        'filename':     filename
    }

    print("Rendering...")
    for _ in range(0, args.batch):
        ws_connect("NEW", json.dumps(data), args.out)


# extract metadata from PNG file
def get_metadata(filepath):
    try:
        im = Image.open(filepath)
    except:
        print("Error: unable to load input file.")
        sys.exit()

    try:
        print(json.loads(im.info["MD"]))
    except:
        print(im.info)
    

if __name__ == '__main__':
    main(sys.argv[1:])
