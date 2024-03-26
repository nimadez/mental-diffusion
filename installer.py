import os
import sys

# Python 3.11.x

# Torch 2.1.2+cu118 is 2.6 GB and you may not be
# able to download it through pip, you can download
# it separately from here:
# https://download.pytorch.org/whl/torch/
# installation:
# python -m pip install torch-2.1.2+cu118-cp311-cp311-win_amd64.whl


def main():
    print("Installing [torch and torchvision]")
    os.system("python -m pip install --no-warn-script-location torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118")
    
    print("Installing [accelerate]")
    os.system("python -m pip install --no-warn-script-location accelerate==0.25.0")
    
    print("Installing [diffusers]")
    os.system("python -m pip install --no-warn-script-location diffusers==0.25.1")
    
    print("Installing [transformers]")
    os.system("python -m pip install --no-warn-script-location transformers==4.36.2")
    
    print("Installing [peft]")
    os.system("python -m pip install --no-warn-script-location peft==0.7.1")
    
    print("Installing [safetensors]")
    os.system("python -m pip install --no-warn-script-location safetensors==0.4.1")
    
    print("Installing [realesrgan]")
    os.system("python -m pip install --no-warn-script-location realesrgan==0.3.0")

    print("Installing [websockets]")
    os.system("python -m pip install --no-warn-script-location websockets==12.0")
    
    print("Installing [omegaconf]")
    os.system("python -m pip install --no-warn-script-location omegaconf==2.3.0")    


if __name__== "__main__":
    if not sys.version.startswith("3.11"):
        print("Python 3.11.x is required.")
        input('Press any key to continue ...')
        sys.exit(0)

    print("Python Package Installer")
    if input("Begin Installation (Y/N)? ").upper() == "Y":
        main()
        print("Finish.")

    input('Press any key to continue ...')
