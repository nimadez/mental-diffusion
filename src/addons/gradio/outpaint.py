#!/usr/bin/env python3
# https://github.com/nimadez/mental-diffusion/


from app import *
addon = Addon()

mdx_outpaint = __import__('mdx-outpaint')


def update_outpaint_aspect(val):
    return val, val, val, val


def download_image(img):
    if img:
        addon.download_png(Image.open(img), "outpaint_image")


def download_mask(img):
    if img:
        addon.download_png(Image.open(img), "outpaint_mask")


def application(app):
    out_image_1 = gr.Image(render=False, label="Image output", type="filepath", sources=["upload"], format="png", height=370, elem_classes=["img_contain"])
    out_image_2 = gr.Image(render=False, label="Mask output", type="filepath", sources=["upload"], format="png", height=370, elem_classes=["img_contain"])

    with gr.Row():
        in_image = gr.Image(label="Input", type="filepath", sources=["upload"], format="png", height=310, elem_classes=["img_contain"])
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    in_pad_t = gr.Number(label="Top", value=100, interactive=True)
                    in_pad_b = gr.Number(label="Bottom", value=100, interactive=True)
                with gr.Row():
                    in_pad_l = gr.Number(label="Left", value=100, interactive=True)
                    in_pad_r = gr.Number(label="Right", value=100, interactive=True)
            in_pad = gr.Number(label="Aspect", value=100, interactive=True)
            in_pad.change(update_outpaint_aspect, inputs=[in_pad], outputs=[in_pad_t, in_pad_b, in_pad_l, in_pad_r])
            
            with gr.Group():
                with gr.Row():
                    btn_create = gr.Button("create images", variant="primary")
                    btn_create.click(mdx_outpaint.create_outpaint, inputs=[in_image, in_pad_t, in_pad_b, in_pad_l, in_pad_r], outputs=[out_image_1, out_image_2])
                    gr.ClearButton(components=[in_image, out_image_1, out_image_2])

    with gr.Group():
        with gr.Row():
            with gr.Column():
                out_image_1.render()
                with gr.Row():
                    btn_image_1_down = gr.Button("download")
                    btn_image_1_down.click(download_image, inputs=[out_image_1])
            with gr.Column():
                out_image_2.render()
                with gr.Row():
                    btn_image_2_down = gr.Button("download")
                    btn_image_2_down.click(download_mask, inputs=[out_image_2])


with gr.Blocks(addon.theme) as app:
    app = addon.init_app("Outpaint", app)
    addon.draw_app(application)


if __name__== "__main__":
    app.launch()
