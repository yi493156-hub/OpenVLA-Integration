import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from reliable_inference_client import (
    ReliableRemoteOpenVLA,
    InferenceHTTPError,
    InferenceNetworkError,
    InferenceResponseError,
)


class TestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)

        if self.path == "/error":
            self.respond(500, {"error": "Model failed"})
        elif self.path == "/invalid":
            self.respond(200, {"action": [0, 0]})
        else:
            self.respond(200, {
                "action": [0.0] * 7,
                "latency_ms": 12.5,
                "backend": "test-double",
                "action_space": "7d_test",
            })

    def respond(self, status, data):
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
thread = threading.Thread(
    target=server.serve_forever,
    daemon=True,
)
thread.start()

image = np.zeros((240, 320, 3), dtype=np.uint8)
base = f"http://127.0.0.1:{server.server_port}"

try:
    client = ReliableRemoteOpenVLA(base + "/predict")
    prediction, latency = client.predict(
        image, "pick up the cube"
    )

    assert prediction.action.shape == (7,)
    assert prediction.backend == "test-double"
    print("[PASS] Valid inference response")

    try:
        ReliableRemoteOpenVLA(base + "/error").predict(
            image, "pick up the cube"
        )
        raise AssertionError("HTTP error was accepted")
    except InferenceHTTPError:
        print("[PASS] HTTP 500 detected")

    try:
        ReliableRemoteOpenVLA(base + "/invalid").predict(
            image, "pick up the cube"
        )
        raise AssertionError("Invalid action was accepted")
    except InferenceResponseError:
        print("[PASS] Invalid response rejected")

    try:
        ReliableRemoteOpenVLA(
            "http://127.0.0.1:1/predict",
            timeout=1,
        ).predict(image, "pick up the cube")
        raise AssertionError("Connection failure was accepted")
    except InferenceNetworkError:
        print("[PASS] Network failure detected")

    print("\nALL RELIABLE INFERENCE CLIENT TESTS PASSED")

finally:
    server.shutdown()
    server.server_close()
