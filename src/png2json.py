#!/usr/bin/env python3
#
# Create a json file from mdx-generated png images
#

import os, sys, json
from PIL import Image

if __name__== "__main__":
    if len(sys.argv) > 1:
        image = sys.argv[1]
        if os.path.exists(image):
            try:
                metadata = json.loads(Image.open(image).info["MDX"])
                output = f"{os.path.dirname(image)}/{os.path.basename(image)}.json"
                with open(output, 'w') as f:
                    f.write(json.dumps(metadata, indent=4))
                print(f"Saved to {output}")
            except:
                print("ERROR: Invalid PNG metadata.")
        else:
            print("ERROR: Image does not exists.")
    else:
        print("help: python3 png2json.py [image]")
