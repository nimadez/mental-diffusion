const data = {
    model: "sd",
    pipeline: "txt2img",
    checkpoint: "/media/$USER/local/ml/stable-diffusion/checkpoints/juggernaut_aftermath.safetensors",
    vae: null,
    lora: null,
    lora_scale: 1.0,
    scheduler: "ddim",
    prompt: "",
    negative: "watermark, ugly, blurry, malformed, disfigured, out of focus",
    width: 512,
    height: 512,
    seed: -1,
    steps: 20,
    guidance: 8.0,
    strength: 1.0,
    image_init: null,
    image_mask: null,
    base64: false,
    filename: "img_{seed}",
    batch: 1,
    preview: true,
    upscale: false
};

const WebSocket = require('ws');
const ws = new WebSocket("ws://localhost:8011/index");

ws.on("open", () => {
    ws.send(JSON.stringify({
        key: "open",
        val: null
    }));

    ws.send(JSON.stringify({
        key: "create",
        val: data
    }));

    ws.close(); // do not close when running batch
});

ws.on("message", (msg) => {
    try {
        //console.log(JSON.parse(msg));
    } catch {
        //console.log(msg);
    }
});

ws.on("close", () => {
    console.log("connection closed");
});

ws.on('error', console.error);
