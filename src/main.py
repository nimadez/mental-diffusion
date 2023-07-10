#
# May 2023
# Mental Diffusion
# @nimadez
#
import os, sys, json, logging
from colorama import Fore


sys.path.append(os.path.abspath("src"))
__import__('colorama').init(autoreset=True)
__import__('warnings').filterwarnings("ignore", category=UserWarning) # disable gfpgan/esrgan torchvision warnings
__import__('transformers').logging.set_verbosity_error() # disable transformers warnings


LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(threadName)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%X")

with open("./config.json", "r") as f:
    config = json.loads(f.read())

os.environ['DISABLE_TELEMETRY'] = "YES"
os.environ['HF_HUB_DISABLE_TELEMETRY'] = "1"
os.environ['HF_HUB_OFFLINE'] = "1" # disable the not-all network calls
os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = "1"
proxy = "Direct"
if config["use_proxy"] == 1:
    os.environ['http_proxy'] = config["proxy"]
    os.environ['https_proxy'] = config["proxy"]
    proxy = config["proxy"]


if __name__ == '__main__':
    import md
    import server
    
    md.preload(config)

    from utils import mem_stats_total
    print(Fore.MAGENTA + "----------------------------------------")
    print(Fore.MAGENTA + f" Mental Diffusion { md.ctx.version }        @nimadez")
    print(Fore.CYAN + f" Torch:        { __import__('torch').__version__ }")
    print(Fore.CYAN + f" Transformers: { __import__('transformers').__version__ }")
    print(Fore.CYAN + f" Diffusers:    { __import__('diffusers').__version__ }")
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
