#
# May 2023
# Mental Diffusion
# @nimadez
#
import os, sys

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning) # disable gfpgan/esrgan torchvision warnings
from transformers import logging as tlog # disable transformers warnings
tlog.set_verbosity_error()

import logging
LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(threadName)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%X")

sys.path.append(os.path.abspath("src"))
import md
import server

localCacheDir = None    # to make a portable huggingface cache

if __name__ == '__main__':
    if localCacheDir:
        os.environ['HF_HOME'] = localCacheDir
        
    md.preload()

    net = "ONLINE"
    if not md.ctx.network:
        net = "OFFLINE"
        os.environ['DISABLE_TELEMETRY'] = "YES"         # disables telemetry collections
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = "1"
        os.environ['HF_HUB_OFFLINE'] = "1"
        os.environ['HF_DATASETS_OFFLINE'] = "1"         # disables all the network calls
        os.environ['TRANSFORMERS_OFFLINE'] = "1"

    from torch import __version__ as ver_troch
    from diffusers import __version__ as ver_diffusers
    print("----------------------------------------")
    print(f" Mental Diffusion { md.ctx.version } - 2023 @nimadez")
    print(f" Torch:         { ver_troch }")
    print(f" Diffusers:     { ver_diffusers }")
    print(f" Device:        { str(md.ctx.device).upper() }")
    print(f" Network:       { net }")
    print(f" Checkpoints:   { len(md.ctx.checkpoints) }")
    print(f" Custom VAE:    { bool(md.ctx.use_VAE) }")
    print("----------------------------------------")

    md.init()

    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        md.clear_pipe()
        server.shutdown()
        sys.exit(0)
