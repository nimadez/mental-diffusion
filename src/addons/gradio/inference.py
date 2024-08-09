#!/usr/bin/env python3
# https://github.com/nimadez/mental-diffusion/


from app import *
addon = Addon()


def inference(*args):
    if args[1]: # -c
        data = addon.json_builder(*args)
        addon.inference(data)


def download_json(*args):
    data = addon.json_builder(*args)
    addon.download_json(data, "inference")


def upload_json(file_path):
    arr = []
    data = json.load(open(file_path))
    for i in data:
        if i == "type":
            data[i] = data[i].upper()
        if i == "image" or i == "mask":
            data[i] = addon.path_normalizer(data[i]) if data[i] else None
        arr.append(data[i])
    return arr


def update_status(p, n, i, m):
    pipe = "error"
    if not i and not m:
        pipe = "txt2img"
    elif i and not m:
        pipe = "img2img"
    elif i and m:
        pipe = "inpaint"
    return f"P [ C: {len(p)}, T: {len(p.split(','))} ], N [ C: {len(n)}, T: {len(n.split(','))} ], PIPE [ {pipe} ]"


def application(app):
    with gr.Row():
        in_type = gr.Radio(['SD', 'XL'], value='XL', info=" ", show_label=False, interactive=True)
        in_checkpoint = gr.Textbox(interactive=True, show_label=False, placeholder='/checkpoint.safetensors', scale=5, lines=1, max_lines=1)
        btn_checkpoint = gr.Button("checkpoint")
        btn_checkpoint.click(addon.file_dialog, inputs=[in_checkpoint], outputs=[in_checkpoint], show_progress="hidden")

    with gr.Group():
        with gr.Row():
            schedulers = ["ddim", "ddpm", "euler", "eulera", "lcm", "lms", "pndm"]
            in_scheduler = gr.Dropdown(schedulers, label="Scheduler", value="euler", interactive=True)
            in_width = gr.Number(label="Width", value=768, minimum=16, interactive=True)
            in_height = gr.Number(label="Height", value=768, minimum=16, interactive=True)
            in_seed = gr.Textbox(label="Seed", value=-1, interactive=True, lines=1, max_lines=1)
        with gr.Row():
            in_steps = gr.Number(label="Steps", value=24, minimum=1, interactive=True)
            in_guidance = gr.Number(label="Guidance", value=8.0, minimum=0, step=0.1, interactive=True)
            in_strength = gr.Number(label="Strength", value=1.0, minimum=0, maximum=1.0, step=0.01, interactive=True)
            in_lorascale = gr.Number(label="LoRA Scale", value=1.0, minimum=0, maximum=1.0, step=0.01, interactive=True)

    with gr.Row():
        in_prompt = gr.Textbox(interactive=True, show_label=False, placeholder="Prompt", lines=5, max_lines=5, elem_classes=["prompts"])
        in_negative = gr.Textbox(interactive=True, show_label=False, placeholder="Negative", lines=5, max_lines=5, elem_classes=["prompts"])
        
    with gr.Row():
        in_image = gr.Image(label="IMAGE", type="filepath", sources=["upload"], format="png", height=120, elem_classes=["img_contain"])
        in_mask = gr.Image(label="MASK", type="filepath", sources=["upload"], format="png", height=120, elem_classes=["img_contain"])
        with gr.Group():
            with gr.Column():
                in_vae = gr.Textbox(interactive=True, label="VAE", placeholder='/vae.safetensors', lines=1, max_lines=1)
                btn_vae = gr.Button("load")
                btn_vae.click(addon.file_dialog, inputs=[in_vae], outputs=[in_vae], show_progress="hidden")
        with gr.Group():
            with gr.Column():
                in_lora = gr.Textbox(interactive=True, label="LoRA", placeholder='/lora.safetensors', lines=1, max_lines=1)
                btn_lora = gr.Button("load")
                btn_lora.click(addon.file_dialog, inputs=[in_lora], outputs=[in_lora], show_progress="hidden")

    with gr.Row():
        in_filename = gr.Textbox(interactive=True, label="Filename", value="img_{seed}", placeholder='File name (img_{seed})', lines=1, max_lines=1)
        in_output = gr.Textbox(interactive=True, label="Output", value="/home/$USER/.mdx", placeholder='Output directory', lines=1, max_lines=1)
        in_number = gr.Number(label="Number", value=1, minimum=1, interactive=True)
        in_batch = gr.Number(label="Batch", value=0, minimum=0, interactive=True)
    with gr.Row():
        out_status = gr.Button("P [ C: 0, T: 1 ], N [ C: 0, T: 1 ], PIPE [ TXT2IMG ]", interactive=False, size="sm", scale=2)
        in_preview = gr.Checkbox(label="Preview", value=True, container=False, scale=1)
        in_lowvram = gr.Checkbox(label="Low VRAM", value=False, container=False, scale=1)

    in_prompt.change(update_status, inputs=[in_prompt, in_negative, in_image, in_mask], outputs=[out_status])
    in_negative.change(update_status, inputs=[in_prompt, in_negative, in_image, in_mask], outputs=[out_status])
    in_image.change(update_status, inputs=[in_prompt, in_negative, in_image, in_mask], outputs=[out_status])
    in_mask.change(update_status, inputs=[in_prompt, in_negative, in_image, in_mask], outputs=[out_status])

    inputs = [
        in_type, in_checkpoint, in_scheduler, in_prompt, in_negative,
        in_width, in_height, in_seed, in_steps, in_guidance, in_strength, in_lorascale,
        in_image, in_mask, in_vae, in_lora,
        in_filename, in_output, in_number, in_batch, in_preview, in_lowvram
    ]

    with gr.Column():
        with gr.Group():
            with gr.Row():
                btn_interrupt = gr.Button("interrupt")
                btn_interrupt.click(addon.interrupt, inputs=[in_output])
                btn_inference = gr.Button("inference", variant="primary")
                btn_inference.click(inference, inputs=inputs)
        with gr.Group():
            with gr.Row():
                btn_open_output = gr.Button("/output", size="sm")
                btn_open_output.click(addon.open_path, inputs=[in_output])
                btn_upload = gr.UploadButton("upload", size="sm", file_types=[".json"])
                btn_upload.upload(upload_json, inputs=[btn_upload], outputs=inputs)
                btn_download = gr.Button("download", size="sm")
                btn_download.click(download_json, inputs=inputs)


with gr.Blocks(addon.theme) as app:
    app = addon.init_app("Inference", app)
    addon.draw_app(application)


if __name__== "__main__":
    app.launch()
