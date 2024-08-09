All notable changes to this project will be documented in this file.

## 0.9.1
- Update diffusers to 0.30.0
- Add image captioning and object detection addons
- Add outpaint and upscaler addons
- Add /libs/realesrgan.py to eliminate the installation (1)
- Allow pip --upgrade for upgradable packages (venv installer)
- Gradio moved to previous experiments (2)
- No major changes to mdx.py

(1) Realesrgan is outdated and relies on numerous unnecessary and incompatible packages, some of which fail to install correctly. To simplify the process, I have included all the required files in a single, self-contained file. The only package that needs to be installed is 'opencv-python'.

(2) I find Gradio to be very good for quick prototyping, but I wouldn't choose it for a more complex user-interface.
