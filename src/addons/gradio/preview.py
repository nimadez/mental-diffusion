#!/usr/bin/env python3
# https://github.com/nimadez/mental-diffusion/


from app import *
addon = Addon()


IMG_NULL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAYAAAD0eNT6AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA+NpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNS1jMDE0IDc5LjE1MTQ4MSwgMjAxMy8wMy8xMy0xMjowOToxNSAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgeG1wTU06T3JpZ2luYWxEb2N1bWVudElEPSJ1dWlkOmM2YmQzYTg3LTkzNmUtNGE1ZS05ODBkLTZmODE5Yzg0ZWNlNCIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDpFNTAwRkMzQTMyRDMxMUVFOUJEOUE4RDUxMEIxMTkwMyIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDpFNTAwRkMzOTMyRDMxMUVFOUJEOUE4RDUxMEIxMTkwMyIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ0MgKFdpbmRvd3MpIj4gPHhtcE1NOkRlcml2ZWRGcm9tIHN0UmVmOmluc3RhbmNlSUQ9InhtcC5paWQ6NGJiMGI3OTktNjE1Mi02MTQzLTkwYjYtNTYzZGYyZTY1ODUzIiBzdFJlZjpkb2N1bWVudElEPSJ4bXAuZGlkOkEwQjhBNEUxNEUzMTExRUNCMTVDRkY3NUYyRjQ0NkY0Ii8+IDxkYzpjcmVhdG9yPiA8cmRmOlNlcT4gPHJkZjpsaT5LZWl0aCBCcm9uaTwvcmRmOmxpPiA8L3JkZjpTZXE+IDwvZGM6Y3JlYXRvcj4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+IDw/eHBhY2tldCBlbmQ9InIiPz4sI0CRAAAQt0lEQVR42uzd7W7iyBaGUTtwt7me3C2xp6MJEopIYog/qva7ltSaX+fMNFTt/YD7Y5zneQAAsrx4CQBAAAAAAgAAEAAAgAAAAAQAACAAAAABAAAIAABAAAAAAgAAEAAAgAAAAAQAACAAAAABAAAIAABAAACAAAAABAAAIAAAAAEAAAgAAEAAAAACAAAQAACAAAAABAAAIAAAAAEAAAgAAEAAAAACAAAQAACAAAAAAQAACAAAQAAAAAIAABAAAIAAAAAEAAAgAAAAAQAACAAAQAAAAAIAABAAAIAAAAAEAAAgAAAAAQAAAgAAEAAAgAAAAAQAACAAAAABAAAIAABAAAAAAgAAEAAAgAAAAAQAACAAAAABAAAIAABAAAAAAgAABAAAIAAAAAEAAAgAAEAAAAACAAAQAACAAAAABAAAIAAAAAEAAAgAAEAAAAACAAAQAADAj85egvW9vb0d9a8eP38MN//8znzzz9m7BlHGOzOj2Vnx+vrqHRMA/HKJl1zme5d7FgRgVjwYAmaFAOCgT/nPXuKfvjW4d7ldcDArvkaEWSEAOOAyv6xwkZcGwfVSTy43mBVmhQBg/8v8slLB/2WQTC43mBULvm0QAgKAFbzsUPGP/Le43GBWCAEBwMaX6NTI4v/ucl+/EQDMCrNCALBSyZ86GjoKH45xGtr/s1zMisaXDW1d6FNn56fFTx+Q8Km/p/ltVggACl3oCv/t4L4d820AAoAiC1QEgAW69OdwNisEAPWKWASAWWFWCADCLrSLDWbFo7PC4wABEPvaVz38IgAsfxEgAAhdkCIALP+lP0cEQMxrnvK6iwB4fk6kfDoWAQIgpujTXnMRAJa/D0YCwOs9ZD7vEgHw2PI3HxEAClcEgOUfIfEbUgHgtRYBYD54Dj4c81cZW0ps/jo71CIALH/fAggAr7MI8DKA5e8Dk8XkMIsAsPyxm7zIpVj+IgAs/8fnptkpABxiEQCWv9mJAHCIRQBY/inzEwHgAIsAsPx9gEIA9HN4vb4iACx/ASAAfPpHBGD5Y44KAAcXEYDlj28BBIDljwjA8gcBIAIQAVj+5qhZKgAcWhEAlj8IAJ/+RQBY/uYpAgARAJY/CABEAFj+IAAQAWD5d8sjAAEAIgDLHwQAIsD5xfJPMHsJBACIACx/EACKFRGA5Q8CABEAlr8PVAiABg6sQysCsPxBAAgARACWP2apAEg5uIgALH/MUgEAIgDLH8tfACQcXIdXBGD5IwAEgABABGD5Y4YKAPWKCMDyRwAIAAcYEYDljw9QAkAAIAKw/M1OBICKRQRg+QsABEBPJgdZBGD58/DcRAA4zIgALH8fmhAADjQiAMu/qtkHJgHgWwBEAJZ/ZgD4sCQASgaACBABWP749C8AQiNA3YoALH/MRwGgcBEBWP6Wv9koABx0RACWf94Ho3cvgwBI8S4CRACWP5a/AMj9JsDzLhGA5Z++/M1BAeDwIwKw/M0/BIBLgAjA8q/KN6ACABEgArD8w+bdZfBroAQAIkAEYPmbcwgAl8PlEAFY/uYbAsAlQQRg+Rfhmb8AQASIACz/sHnmmb8AQASIACx/cwwBgMsjArD8K88vX/sLAESACMDyD5xbvvYXAIgAEYDlb14hAHCpRACWf9U55Wt/AYAIEAFY/oHzydf+AgARIAKw/M0lBAAumwjA8jePEAC4dCIAy78Ez/wFACIAEWD5h80ff8KfAEAEIAIsf3MHAYDLiAiw/M0bBAAuJSLA8i8zZzzzFwCIAESA5R84XzzzFwCIAESA5W+uIABwWREBlr95ggDApUUEWP7mCAIAlxcRYPmbHwgAXGJEgOVvbiAAcJlFQNo9tPzNCwQALjVhEWD5mxMIAFxuwiLA8jcfEAC45IRFgOXf9lzwJ/wJAEQAIsDyD5wH/oQ/AYAIQARY/uYAAgCXHxFg+bv/CAAMAUSA5V+CZ/4IAESACLD8w+77ZfDMHwGACBABlr97jgDAcDAcRIDl734jADAkEAGWf5F77Zk/AgARIAIs/8D77Jk/AgARIAIsf/cYBACGhwiw/KtyfxEAiAARYPm7tyAAMExEgOVf/b565o8AQASIAMvfPQUBgOEiAiz/qvfTb/VDACACRIDlH3gvfe2PAEAEiADL330EAYChIwIs/6r30Nf+CABEAKtHgOXf/v3ztT8CABHAqhFg+bt3CAAwjMIiwPJ33xAAYCiFRYDl3zbP/BEAiABWjwDLv+37dRk880cAIAJYOQIsf/cKAQCGVVgEWP7uEwIADK2wCBgt/6Z55s9mzl4CNo6A0+eSob0IGL03IhrfAIAhlnn/BYB7gwAAwwzcFwQAGGqwN8/8EQCIAAi7H36fPwIAEQDuBQgADDtwH0AAYOiBewACAMMPnH8QABiC0MG596v9EQAYhiKAwPPuV/sjAEAE4JyDAMBwNBxxvkEAYEhCCZ75IwBABBB2nv0JfwgAEAE4xyAAwPDE+QUBAIYoZXjmjwAAEUDYefXMHwEAIgDnFAQAGK44nyAAwJClDM/8EQAgAgg7j575IwBABOAcggAAwxfnDwQAGMI4dyAAwDDGeQMBAIYyHZwzv9ofAQAigMDz5Vf7IwBABOBcgQAAwxrnCQQAGNqU4Jk/AgBEAGHnx5/whwAAEYBzAwIADHOcFxAAYKhT5px45o8AABFA4PnwzB8BACIA5wIEABj2VOU8IABABOAcgAAAw5/q779n/ggAEAF430EAgGVA1ffbb/UDAQAiIPB99rU/CAAQAd5fEABgSVgSVd9XX/uDAAAREPh++tofBACIAO8jIADA8vD+gQAALJESPPMHAQAiIOz9ugye+YMAgBUjwFIRayAAIHC5WCzeIxAAEHhX3BfvEZRx9hLAosVy8jJ04fo+eVwDvgEAyz8wAsw2EABg+YsAQACA5S8CAJcDLH8RAAIAsPxFAAgAsPwRASAAwPJHBIAAAMsfEQACACx/RAAIALD8EQEgAMDyRwSAAADLHxEAAgAsf0QACACw/BEBIADA8kcEgAAAyx8RAAIALH9EAAgAsPwRASAAsPxBBIAAwPIHEQACAMsfRAACACx/EAEIALD8QQQgAMDyBxGAAADLHxEAAgAsf0QACACw/BEBIADA8kcEgAAAyx8RAAIALH9EAAgAsPwRASAAwPJHBIAAwPIHEQACAMsfRAAIACx/EAEgALD8QQSAAMDyBxEAAgDLH0QAAgAsfxABCACw/EEEIADA8gcRgAAAyx9EAAIALH8QAQgAsPxBBCAAwPIHEYAAwPL3MoAIQABg+QMiAAGA5Q8iwMuAAMDyBxEAAgDLn9XM/368//sxeSlEAJnOXgIs/9jl//HP8eYH7UXAINLwDQCWP2u5Lv+vMYBvAhAAYPkHLP9BBIgABABY/rV9LPfL8P3XySJABCAAwPIvuPyXLHcRIAIQAGD5hy1/ESACEABg+Rda/tMTy1wEiAAEAFj+nX/yn/74vxcBIgABAJZ/Z8t/buT/BxGAAMDyZwdrLm0RIAIQAGD5d/DJ/7LBshYBIgABgPNi+Te8/Ldc0iJABCAAsPxpcPk/86v9RYAIQACA5d/5J/9p53+fCBABCAAsfw5e/nPIvxcRgADA8rf8D17CIkAEIACw/DnAHs/8RYAIQABg+dPQJ/+f/lY/EYAIQABg+Rdc/i0uWxEgAhAAWP6ELlkRIAIQAFj+bLBcW3jmLwJEAAIAy5+dl+rU2X+vCBABCAAsf8KWqQgQAQgALH9Cl6gIEAEIACx/QpenCBABCAAsf0KXpggQAQgALH9Cl6UIEAEIACx/QpekCBABCAAsf0KXowgQAQgALH9Cl6IIaD8CzAoBgOXPDsuwhz/hTwSYGQgAXGRWXoJT+M9fBJgdCABcYMvP64AZggDAxbX0vB6YJQgAXNgiEp/5iwAzBQGAixq95C5D7jN/EWC2IABwQS03vE5mDAIAF9NSw+tl1iAAcCGL8cxfBJg5CABcxLAl5pm/CDB7EAC4gJYXXkczCAGAi2dp4fU0ixAAuHBleOYvAswkBAAuWtiS8sxfBJhNCABcMMsJr7MZhQDAxaq8lHztLwIwqwQALlTgMvK1vwjAzBIAuEiWEF5/s8vsEgC4QFWXj6/9RQBmmADAxQlcOr72FwGYZQIAF8aywfuCmSYAcFEsGbw/mG0CABekBM/8RQBmnADAxQhbKv6EPxGAWScAcCEsE7xvmHkCABehMktEBLDd7LN/BECs0etv+SMCgp3MQAGQuvxPn/+kvaXhmb8IQAQIADY79Ja/ZYH3lf/3kHkoACx/LAm8v2Gu34giAMofdK95m8vB7/MXAZiNAoBNP/3T5lLwzF8EcPw+8u2oAHC4sQzwvvsWAAHgYLMNS0AEeP/b20lmpQAoFwA+/bc1/C+Gv3MgApqdlwgAn/4x9HEefAuAAPDpn78Pe7/aHxHgWwAEgNc4cMj71f6IAB+csJwcYsMdnBOzUwCwzSHGUMd5wfwUAF5fduaZPyKg3/kpAiyobuvV4T12iPtb/RABvgVAADi4hjc4R+YoAsDBNbTBeephjpqlAsDyZxHP/BEBIABEQNiQ9swfEeAbAASAQ2s4g3MGAsCnf0MZnDfzFAGAYQzOHQIADGFw/hAAYPiCc9gIjwAEABi6OI8gADBswblMeX0RABiyhgHOJwgAxZr0evoT/hABIAAIHKr+hD9EgA9UCIAuDqxDa5ji3IIAEAAYoji/mKUCIOXg8jzP/BEBmKUCgLDL7m/1QwQwmAMCoOeL79IbluBcIwAEAIYkzjdmqABIOcAs45k/IgABIAAc4LDXyDN/RAA+QAkAAWAYgnPvNfL6CAAVawiC8y8AEAAd8mzb8AP34LHXxSNBAVAqAjD0wH3w6V8ABAaACDDswL3w6V8A+BbAkAPcjztz0owQAMrWcAP3xIxEANSQeLktf3Bflv68EQDlIyCt6C1/EAE/MScEgNIt+PP0lR6IgN+WvzkhAKJqt/KB97U/uEdLZ6Gv/gVAnKqfji1/cJ+W/tx88hcAIsCwAoLulVkhACgUAS40bH+/zApWc/YSNBMBPQeZCw373bOeZ4Vn/gKAbyLg44KfXGig4Kzwu4IEAL8s0+vFHjv4NOK374BZYVYIAFa8LB+l/PJ5sVu83NPgD+6AlmZFq48EzAoBwJMXe2zscit5aHNWTI3NCotfALDS5Z5vYsDiB8wKBECI6eZyjcM+jwbmOz+APmbF9RuBvWeFxS8A2Phyj3d+WPrA128EzAoEQMHLPd/EwHDngo8L/j+GOxfYRQazwqwQAHRywYcvl/z2Qo8LLjSQOSuGX74ZMCsKG+fZewoAafxdAAAgAAAAAQAACAAAQAAAAAIAABAAAIAAAAAEAAAgAAAAAQAACAAAQAAAAAIAABAAAIAAAAAEAAAgAABAAAAAAgAAEAAAgAAAAAQAACAAAAABAAAIAABAAAAAAgAAEAAAgAAAAAQAACAAAAABAAAIAABAAACAAAAABAAAIAAAAAEAAAgAAEAAAAACAAAQAACAAAAABAAAIAAAAAEAAAgAAEAAAAACAAAQAACAAAAAAQAACAAAQAAAAAIAABAAAIAAAAAEAAAgAAAAAQAACAAAQAAAAAIAABAAAIAAAAAEAAAgAAAAAQAAAgAAEAAAgAAAAAQAACAAAAABAAAIAABAAAAAAgAAEAAAgAAAAAQAACAAAAABAAAIAABAAAAAX/wnwACJbTyQ8VtEeAAAAABJRU5ErkJggg=="
img_null = Image.open(BytesIO(b64decode(IMG_NULL.split(',')[1])))


class Watchdog():
    def __init__(self):
        self.last = None
        self.last_img = None
    def change(self, file_path):
        current = os.path.getmtime(file_path)
        if current != self.last:
            self.last = current
            return True
        return False
watch = Watchdog()


def update_preview(outdir):
    outdir = addon.path_normalizer(outdir)
    img = f"{outdir}/preview.jpg"
    if os.path.exists(img):
        if watch.change(img):
            watch.last_img = img
            return img, "Status: updated"
        else:
            return watch.last_img, "Status: no change"
    else:
        watch.last_img = img_null
        return img_null, "Status: not exists"


def update_preview_force(outdir):
    outdir = addon.path_normalizer(outdir)
    img = f"{outdir}/preview.jpg"
    if os.path.exists(img):
        watch.last_img = img
        return img, "Status: updated"
    else:
        watch.last_img = img_null
        return img_null, "Status: not exists"


def create_gallery(outdir):
    paths = []
    outdir = addon.path_normalizer(outdir)
    if os.path.exists(outdir):
        for f in os.listdir(outdir):
            if len(paths) < 100 and f.endswith(".png"):
                paths.append(os.path.join(outdir, f))
        return paths


def save_snapshot(img, outdir):
    if watch.last_img != img_null:
        outdir = addon.path_normalizer(outdir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        img = Image.open(img)
        img.save(f"{outdir}/preview_{random.randrange(100,999)}.png")


def application(app):
    in_output = gr.Textbox(render=False, interactive=True, show_label=False, value="/home/$USER/.mdx", placeholder="Output directory", max_lines=1)
    preview_img = gr.Image(render=False, label="Live Preview", elem_id="preview_img", type="filepath", sources=["upload"], format="png", height=560, elem_classes=["img_contain"])
    preview_stat = gr.Text("Status: undefined", render=False, container=False, max_lines=1)
    out_gallery = gr.Gallery(render=False, label="Output Gallery", object_fit="contain", type="filepath", format="png", columns=4, height=630)
    
    tab_preview = gr.Tab("PREVIEW", render=False)
    tab_gallery = gr.Tab("GALLERY", render=False)
    tab_preview.select(update_preview_force, inputs=[in_output], outputs=[preview_img, preview_stat])
    tab_gallery.select(create_gallery, inputs=[in_output], outputs=[out_gallery])
    app.load(update_preview, inputs=[in_output], outputs=[preview_img, preview_stat], every=1, show_progress="hidden")

    with tab_preview.render():
        preview_img.render()
        with gr.Group():
            with gr.Row():
                preview_stat.render()
                btn_interrupt = gr.Button("interrupt")
                btn_interrupt.click(addon.interrupt, inputs=[in_output])
                btn_preview_update = gr.Button("force update", variant="primary")
                btn_preview_update.click(update_preview_force, inputs=[in_output], outputs=[preview_img, preview_stat])
                btn_preview_capture = gr.Button("snapshot ⤍")
                btn_preview_capture.click(save_snapshot, inputs=[preview_img, in_output])
        in_output.render()
        in_output.change(update_preview_force, inputs=[in_output], outputs=[preview_img, preview_stat], show_progress="hidden")
    
    with tab_gallery.render():
        out_gallery.render()
        with gr.Group():
            with gr.Row():
                btn_open_output = gr.Button("/output")
                btn_open_output.click(addon.open_path, inputs=[in_output])
                btn_gallery_update = gr.Button("force update")
                btn_gallery_update.click(create_gallery, inputs=[in_output], outputs=[out_gallery])


with gr.Blocks(addon.theme) as app:
    app = addon.init_app("Preview", app)
    addon.draw_app(application)


if __name__== "__main__":
    app.launch()
