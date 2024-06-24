//
// Websockets client upscaler
//

const WebSocket = require('ws');
const ws = new WebSocket("ws://localhost:8011/index");

ws.on("open", () => {
    ws.send(JSON.stringify({
        key: "upscale",
        val: {
            "upscale": "x4anime",
            "uri": "tests/mask.png"
        }
    }));

    ws.close();
});

ws.on("message", (msg) => {
    try {
        console.log(JSON.parse(msg));
    } catch {
        console.log(msg);
    }
});

ws.on("close", () => {
    console.log("connection closed");
});

ws.on('error', console.error);
