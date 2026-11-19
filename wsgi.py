"""Production WSGI entry point for the DS-02 Dash dashboard (gunicorn/Render).

Importing this module does not start a development server. It builds the Dash
application and exposes its Flask server object as ``application`` (gunicorn
convention) and ``server``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demand_inventory.dashboard.app import create_app  # noqa: E402

# The Dash application object; its Flask server is app.server.
app = create_app(ROOT)

# Flask WSGI callable served by gunicorn.
application = app.server
server = app.server


@server.route("/health", methods=["GET"])
def health() -> tuple[dict[str, str], int]:
    """Render/application health endpoint.

    Because create_app() loads all required dashboard inputs above, a successful
    import means the dashboard data and model artifacts are present.
    """
    return {"status": "ok"}, 200
