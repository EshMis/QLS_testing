"""Generate HTML (and PDF when Kaleido is available) without launching a UI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qls_testing.core.config import load_config  # noqa: E402
from qls_testing.experiments.run_experiment import run_experiment, run_lindblad_experiment  # noqa: E402
from qls_testing.visualization.plotters import lindblad_figure, trajectory_figure  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/generated_plot.html")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.system.name.startswith("lindblad_"):
        figure = lindblad_figure(run_lindblad_experiment(config))
    else:
        figure = trajectory_figure(run_experiment(config))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output)
    if args.pdf:
        figure.write_image(output.with_suffix(".pdf"))
    print(output)


if __name__ == "__main__":
    main()

