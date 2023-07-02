#
# mental-diffusion asyncio websockets server
#
import logging
log = logging.getLogger("mental-diffusion")
logging.getLogger('websockets.server').setLevel(logging.ERROR)

import os
import sys
import json
import random
import asyncio
import threading
from http import HTTPStatus
from datetime import datetime

from websockets.server import serve

sys.path.append(os.path.abspath("src"))
import md


HOST = "localhost"
PORT = 8011
MAX_SIZE_BYTES = 2 ** 25 # 33MB
threading.current_thread().name = "Main"
records = {} # 'server.log' for previous seasons


def render(data):
    return md.create_image(
        data["scheduler"],
        data["prompt"],
        data["negative"],
        data["width"],
        data["height"],
        data["seed"],
        data["steps"],
        data["cfg"],
        data["strength"],
        data["lora"],
        data["image"],
        data["mask"],
        data["facefix"],
        data["upscale"],
        data["savefile"],
        data["onefile"],
        data["outpath"],
        data["filename"])


async def echo(websocket):
    async for ws in websocket:
        ws = json.loads(ws)
        val = ws["val"]

        match ws["key"]:
            case "RUN": # get startup values
                await websocket.send(json.dumps({
                    "checkpoint": md.ctx.checkpoint_name,
                    "checkpoints": md.ctx.checkpoints
                }))

            case "GET": # get metadata from server by record id
                threading.current_thread().name = val
                if val in records:
                    await websocket.send( json.dumps(records[val]) )
                    log.info("record sent")
                else:
                    await websocket.send("")
                    log.error("record not found")

            case "NEW": # create image
                idx = random.randrange(1000, 9999)
                while idx in records:
                    idx = random.randrange(1000, 9999)
                
                threading.current_thread().name = idx
                log.info("<- [%s]", idx)

                try:
                    arr = render(json.loads(val))
                except KeyboardInterrupt:
                    await websocket.send("")
                    md.clear_cache()
                    log.info("stopped.")
                    return
                
                if arr == None:
                    await websocket.send("")
                    return

                if arr[0] and arr[1]: # [metadata, base64]
                    records[idx] = json.dumps(arr[0])
                    log_file(idx, records[idx])
                    await websocket.send(json.dumps({
                        "metadata": arr[0],
                        "base64":   arr[1]
                    }))
                del arr
                
            case "MOD": # change/reload checkpoint
                threading.current_thread().name = "CKPT"
                try:
                    md.update_checkpoint(val)
                    await websocket.send(val)
                except:
                    await websocket.send("")


def log_file(idx, data):
    with open("server.log", "a") as f:
        f.write(f"id: { idx }\ntime: { datetime.now() }\n{ data }\n\n")


def open_url():
    from webbrowser import open_new_tab
    open_new_tab(f"http://{ HOST }:{ PORT }")


async def request_handler(path, request_headers):
    if path == '/':
        response_headers = [
            ('Server', 'asyncio websocket server'),
            ('Connection', 'close'),
        ]
        body = open("docs/webui/index.html", 'rb').read()
        response_headers.append(('Content-Length', str(len(body))))
        response_headers.append(('Content-Type', "text/html"))
        return HTTPStatus.OK, response_headers, body


async def main():
    log.info(f"running server at http://{ HOST }:{ PORT }")
    log.info("[CTRL+C] terminate running task")
    try:
        server = await serve(
            echo, HOST, PORT,
            max_queue=1, max_size=MAX_SIZE_BYTES,
            process_request=request_handler)
        await server.wait_closed()
    except websockets.exceptions.ConnectionClosed:
        pass


def start():
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()


def shutdown():
    asyncio.get_event_loop().stop()
