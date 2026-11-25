"""Backwards-compatible entry point for a BM25-only benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import main as evaluate_main


def main() -> None:
    """Run the shared evaluator while forcing its retrieval method to BM25."""
    sys.argv[1:] = [*sys.argv[1:], "--methods", "bm25"]
    evaluate_main()


if __name__ == "__main__":
    main()
