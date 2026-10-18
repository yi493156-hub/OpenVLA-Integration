import base64
import io
import json
import time
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from backend import Prediction, validate_image


class RemoteOpenVLA:
    def __init__(self, url="http://127.0.0.1:8765/predict"):
        self.url = url

    def predict(self, image, instruction):
        rgb = validate_image(image)

        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG")

        payload = json.dumps({
            "instruction": instruction,
            "image_base64": base64.b64encode(
                buffer.getvalue()
            ).decode("ascii"),
        }).encode("utf-8")

        request = Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.perf_counter()

        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read())

        round_trip_ms = (time.perf_counter() - start) * 1000

        action = np.asarray(
            result["action"], dtype=np.float32
        ).reshape(-1)

        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise RuntimeError("Invalid action returned by server")

        prediction = Prediction(
            action=action,
            latency_ms=float(result["latency_ms"]),
            backend=result["backend"],
            action_space=result["action_space"],
        )

        return prediction, round_trip_ms
