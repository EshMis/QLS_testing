"""Small truncation-order/step-size sweep using the canonical pipeline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qls_testing.core.config import MethodConfig, load_config  # noqa: E402
from qls_testing.experiments.run_experiment import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/sweep.csv")
    args = parser.parse_args()
    base = load_config(args.config)
    rows = []
    for order in (1, 2, 3):
        for dt in (0.5, 0.25, 0.1):
            if base.time.t_final / dt != round(base.time.t_final / dt):
                continue
            config = replace(base, linearization=MethodConfig("carleman", {"order": order}), time=replace(base.time, dt=dt))
            result = run_experiment(config)
            rows.append({"order": order, "dt": dt, "lifted_dimension": result.metrics["lifted_dimension"], "global_rmse": result.metrics["global_rmse"]})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
