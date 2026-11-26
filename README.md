# Token graph and graph loss

This project builds a graph of tokens that appear in shared contexts and uses it to train token embeddings with an auxiliary triplet loss. It extends [zhasion’s CS336 implementation](https://github.com/zhasion/CS336).

## Where to edit

All paths below are inside `assignment1-basics/`.

| File | What to change |
|---|---|
| `config/graph_experiments.yaml` | Choose the experiments, graph type, context length, loss weight, and training settings. Set `loss_weight: 0` for the baseline. |
| `cs336_basics/graph.py` | Change which tokens share an edge and how edges are counted. |
| `cs336_basics/graph_loss.py` | Change how triplets are sampled or how their loss is calculated. |

## How to run

From the repository root, run `cd assignment1-basics`. For a first run, place `TinyStoriesV2-GPT4-train.txt` and `TinyStoriesV2-GPT4-valid.txt` in `data/full/`, then prepare the data once.

```bash
uv run --python 3.11 python script/prepare_tinystories.py
```

After editing the experiment settings, run the comparison.

```bash
uv run --python 3.11 python run_experiments.py
```

## Where to see the results

Open the newest folder in `exp_output/comparisons/`. Its `loss_comparison.png` shows one curve per experiment, with optimizer updates on the x-axis and validation next-token loss on the y-axis by default. Lower validation loss means better prediction on the evaluated batch.

The `graph_experiments.yaml` beside the plot records the settings that produced it. Each experiment’s subfolder contains its exact `config.yaml` and the loss values in `log.txt`.
