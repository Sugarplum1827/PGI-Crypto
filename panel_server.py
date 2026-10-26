"""Loopback-only presentation server for the PGI-Crypto research system."""
from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .system import PGICryptoResearchSystem

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "panel_static"
DATA_DIR = PACKAGE_DIR / "panel_data"
_system = None
_system_lock = threading.Lock()


def _read_json(filename: str) -> dict:
    with (DATA_DIR / filename).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _panel_summary() -> dict:
    dataset = _read_json("dataset_summary.json")
    comparison = _read_json("model_comparison_summary.json")
    crypto = _read_json("crypto_comparison.json")
    gpu = _read_json("gpu_environment.json")
    strict = _read_json("strict_range_gate.json")
    shap = _read_json("shap_global.json")
    manifest = _read_json("model_manifest.json")

    model_rows = []
    for row in comparison.get("summary", []):
        per_class = row.get("per_class", {})
        mgv = per_class.get("MGV", {})
        model_rows.append({
            "method": row.get("method", ""),
            "model": row.get("model", ""),
            "accuracy": row.get("accuracy_mean"),
            "macro_f1": row.get("macro_f1_mean"),
            "balanced_accuracy": row.get("balanced_accuracy_mean"),
            "log_loss": row.get("log_loss_mean"),
            "fit_seconds": row.get("fit_seconds_mean"),
            "mgv_precision": mgv.get("precision"),
            "mgv_recall": mgv.get("recall"),
            "mgv_support": mgv.get("support"),
            "folds": row.get("folds"),
            "test_rows": row.get("test_rows"),
        })

    crypto_rows = [
        {
            "algorithm": row.get("algorithm"),
            "key_size_bits": row.get("key_size_bits"),
            "trials": row.get("trials"),
            "mean_seconds": row.get("mean_seconds"),
            "median_seconds": row.get("median_seconds"),
            "p95_seconds": row.get("p95_seconds"),
        }
        for row in crypto.get("results", [])
        if row.get("algorithm") in {"RSA-2048", "RSA-4096", "DH-2048", "ECC-P-256"}
    ]

    shap_items = list(zip(shap.get("features", []), shap.get("mean_absolute_shap", [])))
    shap_top = [
        {"feature": name, "mean_absolute_shap": float(value)}
        for name, value in sorted(shap_items, key=lambda item: float(item[1]), reverse=True)[:8]
    ]
    return {
        "project": {
            "name": "PGI-Crypto Research System",
            "system_version": manifest.get("system_version", "1.0.0"),
            "model_track": "adviser_classical_4class",
            "model_features": manifest.get("feature_count"),
            "window_gaps": manifest.get("window_gaps"),
            "window_stride": 50,
            "model_support_min": manifest.get("observed_p_mid_min"),
            "model_support_max": manifest.get("observed_p_mid_max"),
            "model_file": manifest.get("artifact"),
            "model_sha256": manifest.get("sha256"),
        },
        "dataset": {
            "limit": dataset.get("limit"),
            "prime_count": dataset.get("prime_count"),
            "gap_count": dataset.get("gap_count"),
            "bytes": dataset.get("bytes"),
            "max_gap": dataset.get("max_gap"),
            "last_prime": dataset.get("last_prime"),
            "checkpoints_verified": dataset.get("checkpoints_verified", []),
            "window_population": 752_158_239,
            "feature_sample_rows": 376_564,
            "model_evaluation_rows": comparison.get("rows"),
            "model_evaluation_weight_sum": comparison.get("summary", [{}])[0].get("test_weight_sum"),
        },
        "evaluation": {
            "cross_validation": comparison.get("cross_validation"),
            "device_used": comparison.get("device_used"),
            "model_rows": model_rows,
            "chronological_holdout": [
                item for item in comparison.get("statuses", [])
                if "chronological_holdout" in item.get("method", "")
            ],
            "statuses": [
                item for item in comparison.get("statuses", [])
                if item.get("status") != "trained"
            ],
        },
        "hardware": {
            "gpu_name": gpu.get("cuda_device_name"),
            "cuda_device_count": gpu.get("cuda_device_count"),
            "xgboost_cuda_fit": gpu.get("xgboost_cuda_fit"),
            "cupy_kernel_check": gpu.get("cupy_kernel_check"),
            "xgboost_version": gpu.get("xgboost_version"),
        },
        "crypto": {
            "trial_count_per_baseline": crypto.get("trials_per_available_baseline"),
            "baselines": crypto_rows,
            "strict_pgi_status": strict.get("status"),
            "strict_pgi_completed_pairs": strict.get("completed_key_pairs"),
            "strict_pgi_reason": strict.get("reason"),
        },
        "explainability": {
            "method": shap.get("method"),
            "rows_explained": shap.get("data_rows_explained"),
            "top_features": shap_top,
        },
        "research_notice": (
            "This is a research classifier for prime-gap windows. Predictions are not security scores, "
            "do not certify prime gaps, and do not approve cryptographic keys."
        ),
    }


def _example_window() -> dict:
    sample = _read_json("demo_window.json")
    return {
        "p_mid": sample["p_mid"],
        "gaps": sample["gaps"],
        "source": sample["source"],
        "window_id": sample["window_id"],
    }


def _get_system() -> PGICryptoResearchSystem:
    global _system
    if _system is None:
        with _system_lock:
            if _system is None:
                _system = PGICryptoResearchSystem()
    return _system


class PanelHandler(BaseHTTPRequestHandler):
    server_version = "PGICryptoPanel/1.0"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(status, encoded, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        route = urlsplit(self.path).path
        if route == "/api/summary":
            try:
                self._send_json(200, _panel_summary())
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self._send_json(500, {"error": f"Could not read presentation records: {type(exc).__name__}"})
            return
        if route == "/api/example":
            self._send_json(200, _example_window())
            return
        static_files = {
            "/": "index.html",
            "/index.html": "index.html",
            "/styles.css": "styles.css",
            "/app.js": "app.js",
        }
        filename = static_files.get(route)
        if filename is None:
            self._send_json(404, {"error": "Not found"})
            return
        try:
            body = (STATIC_DIR / filename).read_bytes()
        except OSError:
            self._send_json(404, {"error": "Presentation asset is missing"})
            return
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript"}:
            content_type += "; charset=utf-8"
        self._send(200, body, content_type)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if urlsplit(self.path).path != "/api/classify":
            self._send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError("request size is invalid")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = _get_system().classify_window(payload.get("gaps", []), payload.get("p_mid"))
            self._send_json(200, result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(422, {"error": str(exc)})
        except Exception as exc:  # Report a concise error to the local UI; details remain in server logs.
            self.log_error("classification failed: %s", exc)
            self._send_json(500, {"error": f"Classifier could not complete the request ({type(exc).__name__})"})

    def log_message(self, fmt: str, *args) -> None:
        print("[PGI panel] " + (fmt % args))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local PGI-Crypto panel presentation system.")
    parser.add_argument("--host", default="127.0.0.1", help="local interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="local web port (default: 8765)")
    parser.add_argument("--open-browser", action="store_true", help="open the presentation in the default browser")
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        parser.error("the presentation server is restricted to the local computer")

    server = ThreadingHTTPServer((args.host, args.port), PanelHandler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"PGI-Crypto panel system ready: {url}")
    print("Research demo only. PGI predictions do not approve cryptographic keys.")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping PGI-Crypto panel system.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
