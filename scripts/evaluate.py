"""Run BM25, dense, and hybrid comparisons and save JSON plus CSV reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.data.autorag import DEFAULT_DATASET_ID, load_autorag_retrieval
from app.models.embedder import load_embedder
from app.services.benchmark import BenchmarkRunner
from app.services.experiment_service import ExperimentStore

DEFAULTS = Settings()


def parse_args() -> argparse.Namespace:
    """Parse benchmark options and restrict methods to implemented retrievers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default="test")
    parser.add_argument("--revision", default=None, help="Optional immutable Hugging Face revision SHA")
    parser.add_argument("--methods", nargs="+", choices=("bm25", "dense", "hybrid"), default=["bm25"])
    parser.add_argument("--model", default=DEFAULTS.model_name)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments")
    args = parser.parse_args()
    if args.batch_size < 1 or any(k < 1 for k in args.k):
        parser.error("batch-size and K values must be positive")
    if len(set(args.k)) != len(args.k) or len(set(args.methods)) != len(args.methods):
        parser.error("K values and methods must not contain duplicates")
    if any(not 0 <= alpha <= 1 for alpha in args.alpha):
        parser.error("alpha values must be between 0 and 1")
    if len(set(args.alpha)) != len(args.alpha):
        parser.error("alpha values must not contain duplicates")
    return args


def main() -> None:
    """Load the benchmark once, reuse one optional embedder, and persist its results."""
    args = parse_args()
    dataset = load_autorag_retrieval(args.dataset, split=args.split, revision=args.revision)
    embedder = None
    if "dense" in args.methods or "hybrid" in args.methods:
        embedder = load_embedder(args.model, batch_size=args.batch_size, device=args.device)
    report = BenchmarkRunner(
        embedder=embedder,
        index_root=str(ROOT / "data" / "indexes"),
    ).run(dataset, methods=args.methods, k_values=args.k, alpha_values=args.alpha)
    if args.revision:
        report["revision"] = args.revision
    files = ExperimentStore(args.output_dir).save(report)
    print(f"Saved JSON: {files['json']}")
    print(f"Saved CSV:  {files['csv']}")
    for result in report["results"]:
        label = result["method"]
        if result["alpha"] is not None:
            label += f" alpha={result['alpha']}"
        print(f"{label}: {result['metrics']}")


if __name__ == "__main__":
    main()
