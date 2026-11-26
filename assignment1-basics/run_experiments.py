"""Run every entry in a comparison template and plot its loss history."""
import argparse
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml
from matplotlib.figure import Figure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LOSS_TAGS = {"validation": "val loss", "training": "train loss"}


def plot_loss_histories(histories, output_path, ylabel="Validation LM loss"):
    """Plot {experiment name: [(optimizer update, loss), ...]} on one set of axes."""
    if not histories:
        raise ValueError("No experiment histories to plot.")
    figure = Figure(figsize=(8, 4.5), constrained_layout=True)
    axis = figure.subplots()
    markers = ("o", "s", "^", "D", "v")
    for index, (name, history) in enumerate(histories.items()):
        if not history:
            raise ValueError(f"No recorded loss for {name!r}.")
        updates, losses = zip(*history)
        # Stagger markers so curves with nearly identical losses are all visible.
        spacing = min(len(histories), len(history))
        axis.plot(updates, losses, marker=markers[index % len(markers)], markersize=5,
                  markevery=(index % spacing, spacing), label=name)
    axis.set(xlabel="Optimizer update", ylabel=ylabel)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output_path, dpi=150)


def apply_overrides(defaults, overrides):
    """Copy shared settings and replace just the keys specified for this run."""
    settings = deepcopy(defaults)
    for key, value in overrides.items():
        if key not in defaults:
            raise ValueError(f"Unknown experiment setting: {key}")
        if isinstance(defaults[key], dict):
            settings[key] = apply_overrides(defaults[key], value)
        else:
            settings[key] = value
    return settings


def read_loss_history(run_dir, loss_kind):
    events = EventAccumulator(str(run_dir / "tensorboard"), size_guidance={"scalars": 0})
    events.Reload()
    tag = LOSS_TAGS[loss_kind]
    return [(point.step, point.value) for point in events.Scalars(tag)]


def run_comparison(template_path, output_root):
    template_bytes = Path(template_path).read_bytes()
    template = yaml.safe_load(template_bytes)
    loss_kind = template["plot_loss"]
    if loss_kind not in LOSS_TAGS:
        raise ValueError("plot_loss must be 'training' or 'validation'.")
    if not template["experiments"]:
        raise ValueError("Add at least one named experiment to the template.")

    runs = {}
    for name, overrides in template["experiments"].items():
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Choose a simple folder name for experiment {name!r}.")
        config = apply_overrides(template["defaults"], overrides)
        for split in ("train_path", "val_path"):
            path = Path(config["data"][split])
            if not path.is_file():
                raise FileNotFoundError(f"{name}: {path} is missing. Prepare the full splits with "
                                        f"script/prepare_tinystories.py or set data.{split} to an existing token file.")
        runs[name] = config

    # Every invocation saves its own configs, TensorBoard histories, and figure.
    output_dir = Path(output_root) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir.mkdir(parents=True)
    snapshot = output_dir / "graph_experiments.yaml"
    while snapshot.name in runs:
        snapshot = snapshot.with_name("_" + snapshot.name)
    snapshot.write_bytes(template_bytes)
    histories = {}
    for name, config in runs.items():
        run_dir = output_dir / name
        run_dir.mkdir()
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Training {name}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name("train_loop.py")),
                        "--config", str(config_path), "--output-dir", str(run_dir)], check=True)
        histories[name] = read_loss_history(run_dir, loss_kind)

    figure_path = output_dir / "loss_comparison.png"
    plot_loss_histories(histories, figure_path, ylabel=f"{loss_kind.capitalize()} LM loss")
    print(f"Comparison figure: {figure_path.resolve()}")
    return figure_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/graph_experiments.yaml")
    parser.add_argument("--output-dir", default="exp_output/comparisons")
    args = parser.parse_args()
    run_comparison(args.config, args.output_dir)
