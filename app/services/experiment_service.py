"""Safe JSON and CSV storage for completed benchmark runs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


class ExperimentStore:
    """Write experiment reports and retrieve them by generated identifier."""

    def __init__(self, directory: str | Path = "experiments") -> None:
        """Create the output directory lazily when a result is saved."""
        self.directory = Path(directory)

    def save(self, report: dict[str, Any]) -> dict[str, str]:
        """Persist one full JSON report and a flattened CSV row for each method variant."""
        experiment_id = report.get("experiment_id")
        if not isinstance(experiment_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
            raise ValueError("experiment_id must contain only letters, numbers, underscore, or hyphen")
        self.directory.mkdir(parents=True, exist_ok=True)
        json_path = self.directory / f"{experiment_id}.json"
        csv_path = self.directory / f"{experiment_id}.csv"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows: list[dict[str, Any]] = []
        for result in report.get("results", []):
            row = {
                "experiment_id": experiment_id,
                "dataset": report.get("dataset"),
                "split": report.get("split"),
                "revision": report.get("revision"),
                "model": report.get("model"),
                "method": result.get("method"),
                "alpha": result.get("alpha"),
                "query_count": result.get("query_count"),
                "latency_measurement_count": result.get("latency_measurement_count"),
                "latency_mean_ms": result.get("latency_ms", {}).get("mean"),
                "latency_median_ms": result.get("latency_ms", {}).get("median"),
                "latency_p95_ms": result.get("latency_ms", {}).get("p95"),
                "bm25_indexing_seconds": result.get("indexing_seconds", {}).get("bm25"),
                "dense_indexing_seconds": result.get("indexing_seconds", {}).get("dense"),
                "document_encoding_seconds": result.get("document_encoding_seconds"),
            }
            row.update(result.get("metrics", {}))
            rows.append(row)
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return {"json": str(json_path), "csv": str(csv_path)}

    def get(self, experiment_id: str) -> dict[str, Any]:
        """Read a previously saved result after rejecting path traversal identifiers."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
            raise ValueError("invalid experiment ID")
        path = self.directory / f"{experiment_id}.json"
        if not path.is_file():
            raise FileNotFoundError(experiment_id)
        return json.loads(path.read_text(encoding="utf-8"))
