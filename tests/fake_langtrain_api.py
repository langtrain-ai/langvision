"""
An in-process stand-in for the Langtrain API server, with only the routes the
real server serves to API keys (langtrain-server: app/api/v1/finetune.py,
finetune_export.py, files.py, legacy_auth.py). Any other path is a 404, so a
client that calls a route the server doesn't have fails the test.

Keep this in step with the server when routes change.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

API_KEY = "sk-lt-test"
JOB = {
    "id": "job-1", "name": "run", "status": "completed", "progress": 100,
    "config": {"base_model": "meta-llama/Llama-3.1-8B-Instruct"},
    "metrics": {"step": 10, "loss": 0.5}, "error_message": None,
    "created_at": "2026-10-03T00:00:00Z",
}
REQUIRED_JOB_FIELDS = {"base_model", "dataset_id", "training_method"}


class FakeAPI:
    def __init__(self):
        self.calls = []        # (method, path, query, json body or None, api key)
        self.unknown = []      # requests to routes the server doesn't have
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle(self, method):
                url = urlparse(self.path)
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = None
                if raw and "json" in (self.headers.get("Content-Type") or ""):
                    body = json.loads(raw)
                key = self.headers.get("X-API-Key")
                api.calls.append((method, url.path, query, body, key))
                route = (method, url.path)

                if route == ("POST", "/api/v1/auth/api-keys/validate"):
                    if query.get("api_key") != API_KEY:
                        return self._send(401, {"detail": "Invalid API key"})
                    return self._send(200, {"valid": True, "organization_id": "org-1", "plan": "free"})
                if key != API_KEY:
                    return self._send(401, {"detail": "Not authenticated"})
                if route == ("POST", "/api/v1/training/jobs"):
                    missing = REQUIRED_JOB_FIELDS - set(body or {})
                    if missing:
                        return self._send(422, {"detail": f"missing {sorted(missing)}"})
                    return self._send(200, {**JOB, "status": "pending", "progress": 0})
                if route == ("GET", "/api/v1/training/jobs"):
                    if "organization_id" not in query:
                        return self._send(422, {"detail": "organization_id is required"})
                    return self._send(200, {"data": [JOB], "has_more": False})
                if route == ("GET", "/api/v1/training/jobs/job-1"):
                    return self._send(200, JOB)
                if route == ("POST", "/api/v1/training/jobs/job-1/cancel"):
                    return self._send(200, {**JOB, "status": "cancelled"})
                if route == ("POST", "/api/v1/training/jobs/job-1/export"):
                    if not (body or {}).get("repo_id"):
                        return self._send(422, {"detail": "repo_id is required"})
                    return self._send(200, {"export_id": "exp-1", "status": "queued"})
                if route == ("GET", "/api/v1/training/gpu-tiers"):
                    return self._send(200, {"gpu_tiers": [{"id": "t4", "label": "T4 16GB", "vram": 16, "tflops": 65, "price_hr": 0.35, "recommended_for": "Small models"}]})
                if route == ("GET", "/api/v1/training/training-methods"):
                    return self._send(200, [{"id": "qlora"}])
                if route == ("POST", "/api/v1/files"):
                    # The real route needs a dashboard session, not an API key.
                    return self._send(401, {"detail": "Not authenticated"})
                api.unknown.append(route)
                return self._send(404, {"detail": "Not Found"})

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def do_DELETE(self):
                self._handle("DELETE")

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
