#!/usr/bin/env python3
# https://github.com/nimadez/mental-diffusion/


from app import *
addon = Addon()

mdx_upscale = __import__('mdx-upscale')


def inference_upscale(model, image, outdir):
    if image:
        outdir = addon.path_normalizer(outdir)
        mdx_upscale.Upscaler().inference_realesrgan(model.lower(), image, outdir)


def open_last(outdir, img, model):
    if img:
        addon.open_path(f"{outdir}/{os.path.basename(img)}_{model.lower()}.png")


def application(app):
    in_output = gr.Textbox(render=False, interactive=True, label="Output directory", value="/home/$USER/.mdx", placeholder="Output directory", max_lines=1)

    with gr.Row():
        in_image = gr.Image(label="Input", type="filepath", sources=["upload"], format="png", height=512, elem_classes=["img_contain"])

        with gr.Column():
            in_output.render()
            in_model = gr.Radio(['X2','X4'], value='X2', label="Model", interactive=True)
            btn_submit = gr.Button("upscale", variant="primary", scale=1)
            btn_submit.click(inference_upscale, inputs=[in_model, in_image, in_output])
            with gr.Group():
                with gr.Row():
                    btn_open_last = gr.Button("/result")
                    btn_open_last.click(open_last, inputs=[in_output, in_image, in_model])
                    btn_open_output = gr.Button("/output")
                    btn_open_output.click(addon.open_path, inputs=[in_output])
            gr.Examples(examples=addon.examples, inputs=[in_image])


with gr.Blocks(addon.theme) as app:
    app = addon.init_app("Real-ESRGAN Upscaler", app)
    addon.draw_app(application)


if __name__== "__main__":
    app.launch()
