#!/usr/bin/env node
const exec = require('child_process').exec;

const data = {
    type: "xl",
    checkpoint: "/media/$USER/local/ml/sd/ckpt/xl/zavychromaxl_v80.safetensors",
    scheduler: "euler",
    prompt: "a magical dust in a big smoking spherical bottle in alchemists room, colored aura energies, in the style of unreal engine, visual effects, volumetric lighting, bokeh, 8k",
    negative: "watermark, ugly, blurry, malformed, disfigured, out of focus",
    width: 768,
    height: 768,
    seed: -1,
    steps: 28,
    guidance: 7.0,
    strength: 1.0,
    lorascale: 1.0,
    image: null,
    mask: null,
    vae: null,
    lora: null,
    filename: "img_{seed}",
    number: 1,
    batch: 0,
    preview: true,
    lowvram: false
};

function execute(data) {
    const image = (data.image) ? `-i ${data.image}` : '';
    const mask = (data.mask) ? `-m ${data.mask}` : '';
    const vae = (data.vae) ? `-v ${data.vae}` : '';
    const lora = (data.lora) ? `-l ${data.lora}` : '';
    const preview = (data.preview) ? '-pv' : '';
    const lowvram = (data.lowvram) ? '-lv' : '';

    const settings = `${lowvram} ${preview} -t ${data.type} -b ${data.batch} -no ${data.number} -f "${data.filename}"`;
    const inputs = `-c ${data.checkpoint} -sc ${data.scheduler} -p "${data.prompt}" -n "${data.negative}" ${image} ${mask} ${vae} ${lora}`;
    const values = `-w ${data.width} -h ${data.height} -s ${data.seed} -st ${data.steps} -g ${data.guidance} -sr ${data.strength} -ls ${data.lorascale}`;
    
    exec(`x-terminal-emulator -e 'mdx ${settings} ${inputs} ${values}; exec bash;'`);
}

execute(data);
