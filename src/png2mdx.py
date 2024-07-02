#!/usr/bin/env python3
#
# Recreate mdx.py arguments from mdx-generated png images
# (the exact same image is created, you can change the --steps for example)
#
# notice: it does not support images generated with older versions of mdx.py
#

import os, sys, json
from PIL import Image


VENV = "~/.venv/mdx/bin/python3"
ROOT = os.path.dirname(os.path.realpath(__file__))


def create_arguments(data):
    version = data["version"]
    mtype = data["type"]
    checkpoint = data["checkpoint"]
    scheduler = data["scheduler"]
    prompt = data["prompt"]
    negative = data["negative"]
    width = data["width"]
    height = data["height"]
    seed = data["seed"]
    steps = data["steps"]
    guidance = data["guidance"]
    strength = data["strength"]
    lorascale = data["lorascale"]
    image = f'-i {data["image"]}' if data["image"] else ''
    mask = f'-m {data["mask"]}' if data["mask"] else ''
    vae = f'-v {data["vae"]}' if data["vae"] else ''
    lora = f'-l {data["lora"]}' if data["lora"] else ''
    preview = '-pv' if data["preview"] else ''
    lowvram = '-lv' if data["lowvram"] else ''
    a1 = f'-t {data["type"]} -c "{data["checkpoint"]}" -sc "{data["scheduler"]}" -p "{data["prompt"]}" -n "{data["negative"]}" -w {data["width"]} -h {data["height"]}'
    a2 = f'-s {data["seed"]} -st {data["steps"]} -g {data["guidance"]} -sr {data["strength"]} -ls {data["lorascale"]}'
    a3 = f'{image} {mask} {vae} {lora}'
    a4 = f'-f "{data["filename"]}" -o "{data["output"]}" -no {data["number"]} -b {data["batch"]} {preview} {lowvram}'
    return f"{a1} {a2} {a3} {a4}"


if __name__== "__main__":
    if len(sys.argv) > 1:
        image = sys.argv[1]
        if os.path.exists(image):
            try:
                metadata = json.loads(Image.open(image).info["MDX"])
                print('\n', f"{VENV} {ROOT}/mdx.py {create_arguments(metadata)}", '\n')
            except:
                print("ERROR: Invalid PNG metadata.")
        else:
            print("ERROR: Image does not exists.")
    else:
        print("help: python3 png2mdx.py [image]")
