#!/usr/bin/env python3
# https://github.com/nimadez/mental-diffusion/
#
# pip install gradio
# python3 src/addons/argsgen.py

import json
import gradio as gr
from PIL import Image


EXAMPLE_JSON = {
    "type": "xl",
    "checkpoint": "/checkpoint.safetensors",
    "scheduler": "euler",
    "prompt": "prompt",
    "negative": "",
    "width": 768,
    "height": 768,
    "seed": -1,
    "steps": 24,
    "guidance": 8.0,
    "strength": 1,
    "lorascale": 1,
    "image": None,
    "mask": None,
    "vae": None,
    "lora": None,
    "filename": "img_{seed}",
    "output": "/home/$USER/.mdx",
    "number": 1,
    "batch": 0,
    "preview": False,
    "lowvram": False
}


def create_arguments(venv, mdx, data, is_short):
    image = mask = vae = lora = preview = lowvram = ''
    if is_short:
        if "image" in data: image = f'-i "{data["image"]}"' if data["image"] else ''
        if "mask" in data: mask = f'-m "{data["mask"]}"' if data["mask"] else ''
        if "vae" in data: vae = f'-v "{data["vae"]}"' if data["vae"] else ''
        if "lora" in data: lora = f'-l "{data["lora"]}"' if data["lora"] else ''
        if "preview" in data: preview = '-pv' if data["preview"] else ''
        if "lowvram" in data: lowvram = '-lv' if data["lowvram"] else ''
        a1 = f'-t {data["type"]} -c "{data["checkpoint"]}" -sc "{data["scheduler"]}" -p "{data["prompt"]}" -n "{data["negative"]}" -w {data["width"]} -h {data["height"]}'
        a2 = f'-s {data["seed"]} -st {data["steps"]} -g {data["guidance"]} -sr {data["strength"]} -ls {data["lorascale"]}'
        a3 = f'{image} {mask} {vae} {lora}'
        a4 = f'-f "{data["filename"]}" -o "{data["output"]}" -no {data["number"]} -b {data["batch"]} {preview} {lowvram}'
    else:
        if "image" in data: image = f'--image "{data["image"]}"' if data["image"] else ''
        if "mask" in data: mask = f'--mask "{data["mask"]}"' if data["mask"] else ''
        if "vae" in data: vae = f'--vae "{data["vae"]}"' if data["vae"] else ''
        if "lora" in data: lora = f'--lora "{data["lora"]}"' if data["lora"] else ''
        if "preview" in data: preview = '--preview' if data["preview"] else ''
        if "lowvram" in data: lowvram = '--lowvram' if data["lowvram"] else ''
        a1 = f'--type {data["type"]} --checkpoint "{data["checkpoint"]}" --scheduler "{data["scheduler"]}" --prompt "{data["prompt"]}" --negative "{data["negative"]}" --width {data["width"]} --height {data["height"]}'
        a2 = f'--seed {data["seed"]} --steps {data["steps"]} --guidance {data["guidance"]} --strength {data["strength"]} --lorascale {data["lorascale"]}'
        a3 = f'{image} {mask} {vae} {lora}'
        a4 = f'--filename "{data["filename"]}" --output "{data["output"]}" --number {data["number"]} --batch {data["batch"]} {preview} {lowvram}'
    return f"{venv} {mdx} {a1} {a2} {a3} {a4}"


def create_from_png(img, venv, mdx, is_short):
    try:
        metadata = json.loads(Image.open(img).info["MDX"])
        return create_arguments(venv, mdx, metadata, is_short), json.dumps(metadata, indent=4)
    except:
        raise gr.Error("Invalid metadata or not mdx-generated", duration=8)


def create_from_json(js, venv, mdx, is_short):
    try:
        metadata = json.loads(js)
        return create_arguments(venv, mdx, metadata, is_short)
    except:
        raise gr.Error("Invalid JSON data", duration=8)


def fetch_json(inp):
    if inp:
        return json.dumps(inp, indent=4)
    else:
        return json.dumps(EXAMPLE_JSON, indent=4)


with gr.Blocks(theme=gr.themes.Base()) as demo:
    demo.title = "MDX Args Generator"
    demo.analytics_enabled = True
    demo.allow_duplication = True
    demo.allow_flagging = "never"

    gr.Markdown('<h1>MDX Args Generator</h1>')
    
    with gr.Tab("PNG-2-MDX"):
        out_args = gr.TextArea(render=False, label="Command", placeholder='$', show_copy_button=True, autoscroll=False)
        out_json = gr.JSON(render=False, label="JSON output")

        with gr.Row():
            with gr.Column():
                in_image = gr.Image(label="MDX-Generated PNG", type="filepath", sources=["upload"], height=256)
                in_venv = gr.Textbox(show_label=False, value="~/.venv/mdx/bin/python3", placeholder="Python venv/bin", max_lines=1)
                in_mdx = gr.Textbox(show_label=False, value="src/mdx.py", placeholder="Path to mdx.py", max_lines=1)
                in_short = gr.Checkbox(label="Short arguments", value=True)

                with gr.Group():
                    with gr.Row():
                        clear_btn = gr.ClearButton(components=[ in_image, out_args, out_json ], variant="secondary")
                        submit_btn = gr.Button("create", variant="primary")
                        submit_btn.click(create_from_png, inputs=[ in_image, in_venv, in_mdx, in_short ], outputs=[ out_args, out_json ])
                gr.Examples(examples=[ ["media/example.png"] ], inputs=[ in_image ])

            with gr.Column():
                out_args.render()
                out_json.render()
                gr.HTML(value='<div class="info"><b>Reconstructs mdx.py arguments from PNG images generated by mdx.</b><br>- Your image and information will not be saved<br>- The exact same image should be created<br>- It is used for multi-steps image regeneration, content-aware upscaling, and general reconstruction</div>')

    with gr.Tab("JSON-2-MDX"):
        out_args = gr.TextArea(render=False, label="Command", placeholder='$', show_copy_button=True, autoscroll=False)

        with gr.Column():
            with gr.Row():
                in_code = gr.Code(label="JSON Input", value=json.dumps(EXAMPLE_JSON, indent=4), language="json")

            with gr.Row():
                with gr.Column():
                    in_venv = gr.Textbox(show_label=False, value="~/.venv/mdx/bin/python3", placeholder="Python venv/bin", max_lines=1)
                    in_mdx = gr.Textbox(show_label=False, value="src/mdx.py", placeholder="Path to mdx.py", max_lines=1)
                    in_short = gr.Checkbox(label="Short arguments", value=True)

                    with gr.Group():
                        with gr.Row():
                            fetch_btn = gr.Button("fetch", variant="secondary")
                            fetch_btn.click(fetch_json, inputs=[out_json], outputs=[in_code])
                            submit_btn = gr.Button("create", variant="primary")
                            submit_btn.click(create_from_json, inputs=[ in_code, in_venv, in_mdx, in_short ], outputs=[ out_args ])
                    gr.HTML(value='<div class="info">* FETCH: Get the json data from the PNG-2-MDX tab, or reset the json data if none exists.</div>')

                with gr.Column():
                    out_args.render()

    gr.HTML(value="""<div class="info">Theme: <a href="?__theme=light" onclick="const url = new URL(window.location); url.searchParams.set('__theme', 'light'); window.location.href = url.href;">Light</a> | <a href="?__theme=dark" onclick="const url = new URL(window.location); url.searchParams.set('__theme', 'dark'); window.location.href = url.href;">Dark</a></div>""")
    gr.HTML(value='<div class="info">Mental Diffusion: Fast Stable Diffusion CLI<br><a href="https://github.com/nimadez/mental-diffusion/">https://github.com/nimadez/mental-diffusion/</a></div>')

demo.css = """
    a, a:visited { color: steelblue; }
    a:hover { color: #0561ac; }
    h1 { font-size: 20px !important; margin: 0 !important; text-align: center; pointer-events: none !important; }
    h2 { font-size: 14px !important; font-weight: normal !important; color: slategray !important; margin: 5px 0 0 0 !important; text-align: center; pointer-events: none !important; }
    textarea { font-family: monospace; font-size: 12px; }
    button { text-transform: uppercase; }
    .info, b { color: slategray; }
    .cm-content { font-size: 13px; }
    .json-holder { font-size: 13px; }
"""

if __name__ == "__main__":
    demo.launch()
