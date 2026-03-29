"""Server entry point — required by openenv validate."""
from __future__ import annotations

import os
import sys

# Allow imports from repo root when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401  (re-exported for openenv compatibility)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=int(os.getenv("PORT", str(port))))


if __name__ == "__main__":
    main()
