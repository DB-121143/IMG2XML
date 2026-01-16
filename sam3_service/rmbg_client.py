import itertools
import os
import threading
import time
from typing import Dict, List

import requests


class RMBGServiceClient:
    def __init__(self, base_url: str, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = int(os.environ.get("RMBG_TIMEOUT", timeout))

    def health(self) -> bool:
        resp = requests.get(f"{self.base_url}/health", timeout=5)
        return resp.status_code == 200

    def remove(self, image_base64: str) -> str:
        last_exc = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.base_url}/remove",
                    json={"image": image_base64},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["image"]
            except Exception as exc:  # pragma: no cover
                last_exc = exc
                if attempt < 2:
                    time.sleep(1.0)
        raise last_exc


class RMBGServicePool:
    def __init__(self, endpoints: List[str], timeout: int = 60) -> None:
        if len(endpoints) == 0:
            raise ValueError("At least one RMBG endpoint is required")
        self.clients = [RMBGServiceClient(url, timeout=timeout) for url in endpoints]
        self._lock = threading.Lock()
        self._cursor = itertools.cycle(range(len(self.clients)))
        # Limit total in-flight requests to the number of endpoints to avoid piling on one server
        self._sem = threading.Semaphore(len(self.clients))

    def remove(self, image_base64: str) -> str:
        # Ensure we never exceed endpoint count concurrently (across all threads/images)
        self._sem.acquire()
        try:
            with self._lock:
                idx = next(self._cursor)
            return self.clients[idx].remove(image_base64)
        finally:
            self._sem.release()

    def health(self) -> Dict[str, bool]:
        status: Dict[str, bool] = {}
        for client in self.clients:
            try:
                status[client.base_url] = client.health()
            except Exception:
                status[client.base_url] = False
        return status
