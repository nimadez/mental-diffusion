#
# Mental Diffusion Installer
#
import os
import sys
import shutil
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

VERSION = "0.0.77"
MDZIP = "https://github.com/nimadez/mental-diffusion/archive/refs/heads/main.zip"
PYZIP = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
PIPGET = "https://bootstrap.pypa.io/get-pip.py"
EXCLUDE = [ "models", "python_embed", "gfpgan", ".output", "config.json.bkp" ]

config_json = """{
    "use_CPU": 0,
    "use_VAE": 1,

    "use_proxy": 0,
    "proxy": "http://127.0.0.1:8118",

    "checkpoint": "deliberate_v2",
    "checkpoints_root": "models/checkpoints/",

    "vae": "models/vae/vae-ft-mse-840000-ema-pruned.safetensors",
    "gfpgan": "models/gfpgan/GFPGANv1.4.pth",
    "realesrgan": "models/realesrgan/RealESRGAN_x4plus.pth"
}
"""

run_bat = """@echo off
title Mental Diffusion Server
python_embed\python.exe -s src/main.py
pause
"""


def main():
    cwd = os.getcwd()
    DIR_SRC = cwd + "/mental-diffusion/mental-diffusion-main"
    DIR_DST = cwd + "/mental-diffusion"
    DIR_DST_PYTHON = cwd + "/mental-diffusion/python_embed"
    FILE_DST_GETPIP = cwd + "/mental-diffusion/python_embed/get-pip.py"
    FILE_DST_PYTHON_CHECK = cwd + "/mental-diffusion/python_embed/python310.zip"
    FILE_DST_CONFIG = cwd + "/mental-diffusion/config.json"
    FILE_DST_BACKUP = cwd + "/mental-diffusion/config.json.bkp"


    print("-----------------------------------")
    print(" Mental Diffusion Installer " + VERSION)
    print("-----------------------------------")
    opt_update_repo = "Y"
    opt_update_python = "N"

    if os.path.exists(DIR_DST):
        opt_update_repo = input(" Update Mental Diffusion (Y/N)? ").upper()
    else:
        if input(" Begin Installation (Y/N)? ").upper() != "Y":
            sys.exit(0)
    if os.path.exists(FILE_DST_PYTHON_CHECK):
        opt_update_python = input(" Update Python Packages (Y/N)? ").upper()

    print()
    bkp_config = None
    if opt_update_repo == "Y":
        # backup user data
        if os.path.exists(FILE_DST_CONFIG):
            print("\nBacking up user data...")
            with open(FILE_DST_CONFIG, "r") as f:
                bkp_config = f.read()

        # clear previous MD installation
        if os.path.exists(DIR_DST):
            os.chdir(DIR_DST)
            for item in os.listdir(os.getcwd()):
                if item not in EXCLUDE:
                    if os.path.isfile(item):
                        os.remove(item)
                    elif os.path.isdir(item):
                        shutil.rmtree(item, ignore_errors=True)

        if bkp_config:
            with open(FILE_DST_BACKUP, "w") as f:
                f.write(bkp_config)
            print("Backup file created.")

        # download repository
        print('\nDownloading [mental-diffusion]...')
        downloadZip(MDZIP, DIR_DST)
        for f in os.listdir(DIR_SRC):
            shutil.move(os.path.join(DIR_SRC, f), DIR_DST)
        os.rmdir(DIR_SRC)
        print('Done')

        # setup config and directories
        print("\nSetting up config files...")
        with open(FILE_DST_CONFIG, "w") as f:
            f.write(config_json)

        with open(DIR_DST + "/run.bat", "w") as f:
            f.write(run_bat)

        os.chdir(DIR_DST)
        if not os.path.exists(DIR_DST + "/.output"):
            os.makedirs(".output")
        if not os.path.exists(DIR_DST + "/models/checkpoints"):
            os.makedirs("models/checkpoints")
        if not os.path.exists(DIR_DST + "/models/realesrgan"):
            os.makedirs("models/realesrgan")
        if not os.path.exists(DIR_DST + "/models/gfpgan"):
            os.makedirs("models/gfpgan")
        if not os.path.exists(DIR_DST + "/models/loras"):
            os.makedirs("models/loras")
        if not os.path.exists(DIR_DST + "/models/vae"):
            os.makedirs("models/vae")
        print("Done")
    

    # install python and dependencies
    if opt_update_python == "Y" or not os.path.exists(FILE_DST_PYTHON_CHECK):
        if not os.path.exists(FILE_DST_PYTHON_CHECK):
            print('\nInstalling [python 3.10.11 embeddable package]...')
            downloadZip(PYZIP, DIR_DST_PYTHON)
            
        os.chdir(DIR_DST_PYTHON)
        if os.path.exists(DIR_DST_PYTHON + "/python310._pth"):
            os.remove("python310._pth")

        if os.path.exists(FILE_DST_GETPIP) and os.path.getsize(FILE_DST_GETPIP) < 2578580:
            os.remove(FILE_DST_GETPIP)
        if not os.path.exists(FILE_DST_GETPIP):
            print('Downloading [pip]...')
            downloadBin(PIPGET, "get-pip.py")

        if not os.path.exists(DIR_DST_PYTHON + "/Lib/site-packages/pip") or not os.path.exists(DIR_DST_PYTHON + "/Lib/site-packages/setuptools") or not os.path.exists(DIR_DST_PYTHON + "/Lib/site-packages/wheel"):
            print('Installing [pip]...')
            os.system("python get-pip.py")

        print("Installing [torch and torchvision]...")
        os.system("python -m pip install --no-warn-script-location torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118")
        print("Installing [accelerate]...")
        os.system("python -m pip install --quiet --no-warn-script-location accelerate==0.20.3")
        print("Installing [diffusers]...")
        os.system("python -m pip install --quiet --no-warn-script-location diffusers==0.18.1")
        print("Installing [transformers]...")
        os.system("python -m pip install --quiet --no-warn-script-location transformers==4.30.2")
        print("Installing [omegaconf]...")
        os.system("python -m pip install --quiet --no-warn-script-location omegaconf==2.3.0")
        print("Installing [safetensors]...")
        os.system("python -m pip install --quiet --no-warn-script-location safetensors==0.3.1")
        print("Installing [realesrgan]...")
        os.system("python -m pip install --quiet --no-warn-script-location realesrgan==0.3.0")
        print("Installing [gfpgan]...")
        os.system("python -m pip install --quiet --no-warn-script-location gfpgan==1.3.8")
        print("Installing [websockets]...")
        os.system("python -m pip install --quiet --no-warn-script-location websockets==11.0.3")
        print("Done")


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
    print("\nInstallation/update complete.")
    input()
    