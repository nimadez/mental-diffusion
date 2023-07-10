#
# May 2023
# Mental Diffusion
# @nimadez
#
import os, sys, json
from colorama import Fore, init
init(autoreset=True)


from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning) # disable gfpgan/esrgan torchvision warnings
from transformers import logging as tlog # disable transformers warnings
tlog.set_verbosity_error()

import logging
LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(threadName)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%X")


with open("./config.json", "r") as f:
    config = json.loads(f.read())

os.environ['DISABLE_TELEMETRY'] = "YES"         # disables telemetry collections
os.environ['HF_HUB_DISABLE_TELEMETRY'] = "1"
os.environ['HF_HUB_OFFLINE'] = "1"
os.environ['HF_DATASETS_OFFLINE'] = "1"         # disables all the network calls
os.environ['TRANSFORMERS_OFFLINE'] = "1"
proxy = "Direct"
if config["use_proxy"] == 1:
    os.environ['http_proxy'] = config["proxy"]
    os.environ['https_proxy'] = config["proxy"]
    proxy = config["proxy"]


sys.path.append(os.path.abspath("src"))
import md
import server


if __name__ == '__main__':
    md.preload(config)

    from utils import mem_stats_total
    from torch import __version__ as ver_troch
    from transformers import __version__ as ver_transformers
    from diffusers import __version__ as ver_diffusers
    print(Fore.MAGENTA + "----------------------------------------")
    print(Fore.MAGENTA + f" Mental Diffusion { md.ctx.version }        @nimadez")
    print(Fore.CYAN + f" Torch:        { ver_troch }")
    print(Fore.CYAN + f" Transformers: { ver_transformers }")
    print(Fore.CYAN + f" Diffusers:    { ver_diffusers }")
    print(Fore.CYAN + f" Device:       { str(md.ctx.device).upper() }")
    print(Fore.CYAN + f" Checkpoints:  { len(md.ctx.checkpoints) }")
    print(Fore.CYAN + f" Custom VAE:   { bool(md.ctx.use_VAE) }")
    print(Fore.CYAN + f" Proxy:        { proxy }")
    print(Fore.CYAN + f" Memory:       { mem_stats_total()[0]:.1f}GB RAM | { mem_stats_total()[1]:.1f}GB VRAM")
    print(Fore.MAGENTA + "----------------------------------------")
    del config
    
    md.init()

    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        md.clear_pipe()
        server.shutdown()
        sys.exit(0)
