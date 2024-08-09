#!/usr/bin/env python3
# https://github.com/nimadez/mental-diffusion/


from app import *
addon = Addon()


from inference import application as inference
from metadata import application as metadata
from outpaint import application as outpaint
from preview import application as preview
IS_REALESRGAN = True
try:
    from upscale import application as upscale
except:
    IS_REALESRGAN = False


def application(app):
    with gr.Tab("INFERENCE"):
        inference(app)
    with gr.Tab("PREVIEW"):
        preview(app)
    with gr.Tab("METADATA"):
        metadata(app)
    with gr.Tab("OUTPAINT"):
        outpaint(app)
    if IS_REALESRGAN:
        with gr.Tab("UPSCALE"):
            upscale(app)


with gr.Blocks(addon.theme) as app:
    app = addon.init_app("Mental Diffusion Addons", app)
    addon.draw_app(application)


if __name__== "__main__":
    app.launch()
