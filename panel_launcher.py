"""Launch the Streamlit panel from the installed project command."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    working_root = Path.cwd()
    local_app = working_root / "src" / "python" / "pgi_crypto" / "streamlit_app.py"
    if local_app.is_file():
        project_root = working_root
        app_file = local_app
    else:
        app_file = Path(__file__).resolve().with_name("streamlit_app.py")
        project_root = app_file.parents[3]
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_file),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.call(command, cwd=project_root)


if __name__ == "__main__":
    raise SystemExit(main())
