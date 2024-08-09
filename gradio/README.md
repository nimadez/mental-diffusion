## Mental Diffusion Gradio Addons

<img src="https://raw.githubusercontent.com/nimadez/mental-diffusion/main/gradio/media/screenshot.png" height="512">

#### Gradio 4.37.2

#### Install Gradio:
```
~/.venv/mdx/bin/python3 -m pip install gradio==4.37.2
```
#### Install Zenity for inference addon:
```
sudo apt install zenity

# Without Zenity, you can't select safetensors files with the file dialog, you have to enter the Checkpoint, VAE and LoRA path manually.
```

| Name | Description |
| --- | --- |
| main | A tabbed interface for all addons |
| inference | The inference user-interface |
| preview | Watch preview and gallery |
| metadata | View and recreate data from PNG |
| outpaint | Create image and mask for outpaint |
| upscale | Real-ESRGAN x2 and x4 plus |

```
cd gradio # important
~/.venv/mdx/bin/python3 addon-name.py
~/.venv/mdx/bin/python3 main.py
```

> The current directory structure is required to connect and work with mdx.py and addon scripts.

## License
Code released under the [MIT license](https://github.com/nimadez/mental-diffusion/blob/main/LICENSE).

## Credits
- [Gradio](https://www.gradio.app/)
