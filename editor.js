#!/usr/bin/env node
// JavaScript to mdx.py arguments
// apt install nodejs
// node editor.js

const VENV = "~/.venv/mdx/bin/python3";

const data = {
    type: "xl",
    checkpoint: "/media/$USER/local/ml/sd/ckpt/xl/zavychromaxl_v80.safetensors",
    scheduler: "euler",
    prompt: "a magical dust in a big smoking spherical bottle in alchemists room, colored aura energies, in the style of unreal engine, visual effects, volumetric lighting, bokeh, 8k",
    negative: "watermark, ugly, blurry, malformed, disfigured, out of focus",
    width: 768,
    height: 768,
    seed: -1,
    steps: 24,
    guidance: 8.0,
    strength: 1.0,
    lorascale: 1.0,
    image: null,
    mask: null,
    vae: null,
    lora: null,
    filename: "img_{seed}",
    output: "/home/$USER/.mdx",
    number: 1,
    batch: 0,
    preview: true,
    lowvram: false
};

function execute(data) {
    const image = (data.image) ? `-i "${data.image}"` : '';
    const mask = (data.mask) ? `-m "${data.mask}"` : '';
    const vae = (data.vae) ? `-v "${data.vae}"` : '';
    const lora = (data.lora) ? `-l "${data.lora}"` : '';
    const preview = (data.preview) ? '-pv' : '';
    const lowvram = (data.lowvram) ? '-lv' : '';

    const a1 = `-t ${data.type} -c "${data.checkpoint}" -sc "${data.scheduler}" -p "${data.prompt}" -n "${data.negative}"`;
    const a2 = `-w ${data.width} -h ${data.height} -s ${data.seed} -st ${data.steps} -g ${data.guidance} -sr ${data.strength} -ls ${data.lorascale}`;
    const a3 = `${image} ${mask} ${vae} ${lora}`;
    const a4 = `-f "${data.filename}" -o "${data.output}" -no ${data.number} -b ${data.batch} ${preview} ${lowvram}`;
    const cmd = `${VENV} src/mdx.py ${a1} ${a2} ${a3} ${a4}`;

    const exec = require('child_process').exec;
    exec(`x-terminal-emulator -e '${cmd}; exec bash;'`);
    return cmd;
}

console.log(execute(data));
