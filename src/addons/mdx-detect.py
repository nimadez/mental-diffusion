#!/usr/bin/env python3
# Aug 2024 | Mental Diffusion | https://github.com/nimadez/mental-diffusion
#
# Transformers object detection (DETR and YOLO methods)
#
# Installation:
# $ pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
# $ pip install transformers
#
# Inference:
# python3 mdx-detect.py --help
# python3 mdx-detect.py -i https://example.com/image.png
# python3 mdx-detect.py -i ./image.png
# python3 mdx-detect.py -i ./image.png -t 0.5
# python3 mdx-detect.py -i ./image.png -t 0.5 -p 10
# python3 mdx-detect.py -i ./image.png --yolo
# python3 mdx-detect.py -i ./image.png --view
# python3 mdx-detect.py -i ./image.png -o ~/Downloads
#
# - CPU and GPU
# - Support all aspect ratios
# - Add extra padding per object
# - Process images from https links
# - YOLO detection is almost real-time
# - If --output is defined, extract and save images


MODEL_DETR = "facebook/detr-resnet-50"
MODEL_YOLO = "hustvl/yolos-tiny" # faster
THRESHOLD = 0.6


import os, sys, gc, time, random, requests
import torch
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers import DetrImageProcessor, DetrForObjectDetection


def arg_parser(args):
    parser = ArgumentParser("mdx-detect.py", add_help = False)
    parser.add_argument('--help', action = "help", help = "show this help message and exit")
    parser.add_argument('-i', '--image', type = str, default = None, help = "absolute or relative image path or https:// url")
    parser.add_argument('-t', '--threshold', type = float, default = THRESHOLD, required = False, help = f"float number between 0.1 and 0.9 (def: {THRESHOLD})")
    parser.add_argument('-p', '--padding', type = int, default = 0, required = False, help = f"integer number, add padding to objects (def: 0)")
    parser.add_argument('-o', '--output', type = str, default = None, help = "optional output directory to save extracted images (def: None)")
    parser.add_argument('-y', '--yolo', action = 'store_true', help = "use faster YOLO method (def: no yolo)")
    parser.add_argument('-v', '--view', action = 'store_true', help = "display the preview window (def: no view)")
    return parser.parse_args(args)


class ObjectDetection():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_processor = None
        self.model = None
        self.data = []


    def load_model(self, use_yolo=True):
        if use_yolo:
            self.image_processor = AutoImageProcessor.from_pretrained(MODEL_YOLO)
            self.model = AutoModelForObjectDetection.from_pretrained(MODEL_YOLO)
        else:
            self.image_processor = DetrImageProcessor.from_pretrained(MODEL_DETR, revision="no_timm")
            self.model = DetrForObjectDetection.from_pretrained(MODEL_DETR, revision="no_timm")
        self.model.to(self.device)


    def processor(self, image, threshold, padding):
        image = image.convert("RGB")
        w, h = image.size   # only 1:1 aspect is allowed
        nw, nh = image.size # here we fix the dimensions
        img = image
        if w != h:
            size = max([w, h])
            img = Image.new('RGB', (size, size), (0,0,0))
            nw, nh = img.size
            img.paste(image, (int(nw/2)-int(w/2), int(nh/2)-int(h/2)), image.convert('RGBA'))

        inputs = self.image_processor(images=[img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([img.size[::-1]])
            results = self.image_processor.post_process_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=target_sizes)[0]

        self.data = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x, y, x2, y2 = tuple(box)
            if padding > 0:
                if x  - padding > 0:  x  -= padding
                if y  - padding > 0:  y  -= padding
                if x2 + padding < nw: x2 += padding
                if y2 + padding < nh: y2 += padding
            self.data.append({
                "index": len(self.data),
                "score": round(score.item(), 3),
                "label": self.model.config.id2label[label.item()],
                "x": x, "y": y, "x2": x2, "y2": y2 })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        return img


    def extract(self, image, filename, outdir):
        for obj in self.data:
            img = image.crop((obj["x"], obj["y"], obj["x2"], obj["y2"]))
            img.save(f"{outdir}/{filename}_{obj["index"]}_{obj["score"]}_{obj["label"].replace(' ', '_')}.png")


    def reveal(self, image):
        draw = ImageDraw.Draw(image, "RGBA")
        for obj in self.data:
            c = random.choices(range(256), k=3)
            c = (c[0], c[1], c[2], 128)
            draw.rectangle((obj["x"], obj["y"], obj["x2"], obj["y2"]), outline=c, width=2)
            draw.rectangle((obj["x"], obj["y"]-18, obj["x2"], obj["y"]), fill=c)
            draw.text((obj["x"]+5, obj["y"]-16), f"#{obj["index"]} {obj["label"].upper()}", font_size=12, fill="white")
        return image


    def printer(self):
        for obj in self.data:
            print(f" {"{:02.0f}".format(obj["index"])}", f"| {format(obj["score"], '.3f')} |", f"{obj["label"].upper()}:", f"[{obj["x"]}, {obj["y"]}, {obj["x2"]}, {obj["y2"]}]")


if __name__== "__main__":
    if len(sys.argv) > 1:
        args = arg_parser(sys.argv[1:])

        image = args.image
        if os.path.exists(image) or image.startswith("https://"):
            if image.startswith("https://"):
                r = requests.get(image)
                if r.status_code == 200:
                    image = Image.open(BytesIO(r.content))
                else:
                    print('ERROR: Image is not downloaded, invalid URL.')
                    sys.exit()
            else:
                image = Image.open(image)

            od = ObjectDetection()
            print(f"Device: {od.device.upper()}")
            print(f"Loading {MODEL_YOLO if args.yolo else MODEL_DETR} ...")
            od.load_model(args.yolo)

            print("Processing image ...")
            start = time.time()
            image = od.processor(image, args.threshold, args.padding)
            print(f"Process time: {round(time.time() - start, 3)} seconds")

            od.printer()
            draw = od.reveal(image)

            if args.output:
                if os.path.exists(args.output):
                    draw.save(f"{args.output}/{Path(args.image).stem}_detects.png")
                    od.extract(image, Path(args.image).stem, args.output)
                    print(f"Saved to {os.path.abspath(args.output)}")
                else:
                    print("ERROR: Output directory does not exist.")

            if args.view:
                draw.show()
        else:
            print("ERROR: Image does not exists.")
    else:
        print("help: python3 mdx-detect.py --help")
