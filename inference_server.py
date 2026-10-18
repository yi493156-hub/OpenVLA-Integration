import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from backend import MockOpenVLA, RealOpenVLA


def create_handler(backend):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 5_000_000:
                    raise ValueError("Invalid request size")

                request = json.loads(
                    self.rfile.read(length).decode("utf-8")
                )

                instruction = request["instruction"]
                image_bytes = base64.b64decode(
                    request["image_base64"],
                    validate=True,
                )

                if len(image_bytes) > 3_000_000:
                    raise ValueError("Image too large")

                with Image.open(io.BytesIO(image_bytes)) as img:
                    image = np.asarray(
                        img.convert("RGB"), dtype=np.uint8
                    )

                prediction = backend.predict(image, instruction)

                response = {
                    "action": prediction.action.tolist(),
                    "latency_ms": prediction.latency_ms,
                    "backend": prediction.backend,
                    "action_space": prediction.action_space,
                }

                self.respond(200, response)

            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception as exc:
                self.respond(500, {"error": str(exc)})

        def respond(self, status, data):
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["mock", "real"], default="mock"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    backend = (
        MockOpenVLA()
        if args.backend == "mock"
        else RealOpenVLA()
    )

    server = ThreadingHTTPServer(
        (args.host, args.port),
        create_handler(backend),
    )

    print(
        f"OpenVLA server ready: http://{args.host}:{args.port}",
        flush=True,
    )
    print(f"Backend: {args.backend}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
