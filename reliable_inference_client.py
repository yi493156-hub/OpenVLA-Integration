import base64
import io
import json
import math
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from backend import Prediction, build_prompt, validate_image


class InferenceError(RuntimeError):
    pass


class InferenceHTTPError(InferenceError):
    pass


class InferenceNetworkError(InferenceError):
    pass


class InferenceResponseError(InferenceError):
    pass


class ReliableRemoteOpenVLA:
    def __init__(
        self,
        url="http://127.0.0.1:8765/predict",
        timeout=120,
    ):
        if not isinstance(timeout, (int, float)):
            raise ValueError("Timeout must be numeric")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite")

        self.url = url
        self.timeout = float(timeout)

    def predict(self, image, instruction):
        rgb = validate_image(image)
        build_prompt(instruction)

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

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(1_000_001)

        except HTTPError as exc:
            detail = exc.read(4096).decode(
                "utf-8", errors="replace"
            )
            raise InferenceHTTPError(
                f"HTTP {exc.code}: {detail}"
            ) from exc

        except (URLError, socket.timeout, TimeoutError, OSError) as exc:
            raise InferenceNetworkError(
                f"Inference server unavailable: {exc}"
            ) from exc

        round_trip_ms = (time.perf_counter() - start) * 1000

        if len(raw) > 1_000_000:
            raise InferenceResponseError("Response too large")

        try:
            result = json.loads(raw)

            if not isinstance(result, dict):
                raise ValueError("Response must be an object")

            raw_action = result["action"]
            if not isinstance(raw_action, list) or len(raw_action) != 7:
                raise ValueError("Action must be a list of 7 numbers")

            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                for value in raw_action
            ):
                raise ValueError("Action contains nonnumeric values")

            action = np.asarray(raw_action, dtype=np.float32)

            if not np.all(np.isfinite(action)):
                raise ValueError("Action contains nonfinite values")

            latency = result["latency_ms"]
            if isinstance(latency, bool) or not isinstance(
                latency, (int, float)
            ):
                raise ValueError("Invalid latency")

            latency = float(latency)
            if not math.isfinite(latency) or latency < 0:
                raise ValueError("Invalid latency")

            backend = result["backend"]
            action_space = result["action_space"]

            if not isinstance(backend, str) or not backend:
                raise ValueError("Invalid backend")

            if not isinstance(action_space, str) or not action_space:
                raise ValueError("Invalid action space")

        except (
            ValueError,
            KeyError,
            TypeError,
            OverflowError,
        ) as exc:
            raise InferenceResponseError(
                f"Invalid inference response: {exc}"
            ) from exc

        prediction = Prediction(
            action=action,
            latency_ms=latency,
            backend=backend,
            action_space=action_space,
        )

        return prediction, round_trip_ms
