#!/usr/bin/env python3
# Aug 2024 | Mental Diffusion | https://github.com/nimadez/mental-diffusion
#
# Create outpaint image and mask for inpaint
#
# python3 mdx-outpaint.py ./image.png 200 200 100 100
# python3 mdx-outpaint.py ./image.png 200 200 100 100 ~/Downloads


import os, sys
from pathlib import Path
from PIL import Image, ImageOps


def create_outpaint(image, pad_t, pad_b, pad_l, pad_r):
    if image:
        pad_t = int(pad_t)
        pad_b = int(pad_b)
        pad_l = int(pad_l)
        pad_r = int(pad_r)

        img = Image.open(image)
        w, h = img.size

        w += pad_l + pad_r
        h += pad_t + pad_b
        w, h = round(w / 8) * 8, round(h / 8) * 8
        dim = ImageOps.pad(img, (w - (pad_l+pad_r), h - (pad_t+pad_b))).size

        mask_bg = Image.new('RGB', (w, h), (255,255,255))
        mask_fore = Image.new('RGBA', dim, (0,0,0))
        mask_bg.paste(mask_fore, (pad_l, pad_t), mask_fore)

        img_bg = Image.new('RGB', (w, h), (0,0,0))
        img_fore = img
        img_fore.resize(dim)
        img_bg.paste(img_fore, (pad_l, pad_t), img_fore.convert('RGBA'))

        return img_bg, mask_bg
    else:
        return None, None


if __name__== "__main__":
    if len(sys.argv) > 5:
        if os.path.exists(sys.argv[1]):
            image, mask = create_outpaint(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
            if image and mask:
                fname = Path(sys.argv[1]).stem
                if len(sys.argv) == 7:
                    if os.path.exists(sys.argv[6]):
                        image.save(f"{sys.argv[6]}/{fname}_outpaint.png")
                        mask.save(f"{sys.argv[6]}/{fname}_outpaint_mask.png")
                        print(f"Saved to {os.path.abspath(sys.argv[6])}")
                    else:
                        print("ERROR: Output directory does not exist.")
                else:
                    image.save(f"./{fname}_outpaint.png")
                    mask.save(f"./{fname}_outpaint_mask.png")
                    print(f"Saved to {os.path.abspath(".")}")
        else:
            print("ERROR: Image does not exist.")
    else:
        print("help: python3 mdx-outpaint.py [image] [top] [bottom] [left] [right] [directory?]")
