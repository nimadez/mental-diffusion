#
# Mental Diffusion Installer
# Version 0.0.6
#
import os
import sys
import shutil
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


MDZIP = "https://github.com/nimadez/mental-diffusion/archive/refs/heads/main.zip"
PYZIP = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
PIPGET = "https://bootstrap.pypa.io/get-pip.py"
EXCLUDE = ["models", "python_embed", "gfpgan", ".output"]

config_json = """{
    "checkpoints_root": "models/checkpoints/",
    "vae": "models/vae/vae-ft-mse-840000-ema-pruned.safetensors",
    "gfpgan": "models/gfpgan/GFPGANv1.4.pth",
    "realesrgan": "models/upscale/RealESRGAN_x4plus.pth",
    
    "checkpoint": "deliberate_v2",
    "use_CPU": 0,
    "use_VAE": 1
}
"""

run_bat = """@echo off
title Mental Diffusion Server
python_embed\python.exe -s src/main.py
pause
"""

importsite = """python310.zip
.
import site
"""


def main():
    cwd = os.getcwd()
    DIR_SRC = cwd + '/mental-diffusion/mental-diffusion-main'
    DIR_DST = cwd + '/mental-diffusion'
    DIR_DST_PYTHON = cwd + '/mental-diffusion/python_embed'
    DIR_DST_CONFIG = cwd + '/mental-diffusion/config.json'
    DIR_DST_BACKUP = cwd + '/mental-diffusion/config.json.bkp'

    print('----------------------------------')
    print(' Mental Diffusion Installer 0.0.6')
    print('----------------------------------')
    opt_start = input(" Begin installation (Y/N)? ").upper()
    if opt_start == "N":
        sys.exit(0)


    # backup user data
    bkp_config = None
    if os.path.exists(DIR_DST_CONFIG):
        print("\nBacking up user data...")
        with open(DIR_DST_CONFIG, "r") as f:
            bkp_config = f.read()
        print("Backup file created.")

    # clear previous MD installation
    if os.path.exists(DIR_DST):
        os.chdir(DIR_DST)
        for item in os.listdir(os.getcwd()):
            if item not in EXCLUDE:
                if os.path.isfile(item):
                    os.remove(item)
                elif os.path.isdir(item):
                    shutil.rmtree(item, ignore_errors=True)


    print('\nDownloading mental-diffusion repository...')
    downloadZip(MDZIP, DIR_DST)
    for f in os.listdir(DIR_SRC):
        shutil.move(os.path.join(DIR_SRC, f), DIR_DST)
    os.rmdir(DIR_SRC)
    print('Done\n')


    print("Setting up config files...")
    with open(DIR_DST_CONFIG, "w") as f:
        f.write(config_json)
    if bkp_config:
        with open(DIR_DST_BACKUP, "w") as f:
            f.write(bkp_config)

    with open(DIR_DST + "/run.bat", "w") as f:
        f.write(run_bat)

    os.chdir(DIR_DST)
    if not os.path.exists(DIR_DST + "/.output"):
        os.makedirs(".output")
    if not os.path.exists(DIR_DST + "/models"):
        os.makedirs("models/checkpoints")
        os.makedirs("models/gfpgan")
        os.makedirs("models/loras")
        os.makedirs("models/upscale")
        os.makedirs("models/vae")
    print("Done\n")
    

    if not os.path.exists(DIR_DST_PYTHON):
        print('Downloading Python 3.10.11 embeddable package...')
        downloadZip(PYZIP, DIR_DST_PYTHON)
        
    os.chdir(DIR_DST_PYTHON)

    if not os.path.exists(DIR_DST_PYTHON + "/get-pip.py"):
        print('Downloading Python PIP...')
        downloadBin(PIPGET, "get-pip.py")
        os.system("python get-pip.py")

        with open("python310._pth", "w") as f:
            f.write(importsite)
        print("Done\n")

        print('Installing Python packages...')
        packageInstaller()
        print("Done\n")


    opt_force = input(" Force-check installed Python packages (Y/N)? ").upper()
    if opt_force == "Y":
        print('Checking Python packages...')
        packageInstaller()
        print("Done\n")


def packageInstaller():
    os.system("python -m pip install --no-warn-script-location accelerate==0.20.3")
    os.system("python -m pip install --no-warn-script-location diffusers==0.17.1")
    os.system("python -m pip install --no-warn-script-location torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118")
    os.system("python -m pip install --no-warn-script-location transformers==4.30.0")
    os.system("python -m pip install --no-warn-script-location omegaconf==2.3.0")
    os.system("python -m pip install --no-warn-script-location safetensors==0.3.1")
    os.system("python -m pip install --no-warn-script-location realesrgan==0.3.0")
    os.system("python -m pip install --no-warn-script-location gfpgan==1.3.8")
    os.system("python -m pip install --no-warn-script-location websockets")


def downloadBin(url, filepath):
    with urlopen(url) as dat:
        with open(filepath, "wb") as f:
            f.write(dat.read())
            

def downloadZip(url, destdir):
    with urlopen(url) as zip:
        with ZipFile(BytesIO(zip.read())) as zf:
            zf.extractall(destdir)


if __name__== "__main__":
    main()
    print("Installation or update complete.")
    input()
    