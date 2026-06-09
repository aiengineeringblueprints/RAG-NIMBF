"""Minimal HTTP RAG endpoint compatible with the benchmark HTTP adapter.

Run:
    python examples/http_rag_server.py

Then in another shell:
    RAG_SYSTEM_ADAPTER=http \
    RAG_HTTP_ENDPOINT_URL=http://localhost:8000/query \
    DATASET_NAME=jsonl \
    DATASET_PATH=examples/sample_dataset.jsonl \
    RAGAS_ENABLED=false \
    CUSTOM_METRICS_ENABLED=false \
    python main.py
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class RagHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/query":
            self.send_error(404, "Use POST /query")
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        question = payload.get("question", "")
        started = time.perf_counter()

        answer = f"Demo answer for: {question}"
        response = {
            "answer": answer,
            "contexts": [
                "This is a demo context returned by examples/http_rag_server.py."
            ],
            "metadata": [{"doc_id": "demo-doc-1", "score": 1.0}],
            "timings": {
                "ttft_seconds": 0.0,
                "total_seconds": time.perf_counter() - started,
                "token_count": len(answer.split()),
            },
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), RagHandler)
    print("Example RAG server listening on http://127.0.0.1:8000/query")
    server.serve_forever()
