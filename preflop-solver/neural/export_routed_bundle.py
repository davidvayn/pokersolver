#!/usr/bin/env python3
"""Export a frozen street-routed MLX policy as the Rust inference bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlx.core as mx

from train import INPUT_FEATURE_COUNT, NETWORK_SCHEMA, scorer_json
from validate_seeds import load_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflop-run", type=Path, required=True)
    parser.add_argument("--preflop-round", type=int, required=True)
    parser.add_argument("--preflop-weights", type=Path, required=True)
    parser.add_argument("--postflop-run", type=Path, required=True)
    parser.add_argument("--postflop-round", type=int, required=True)
    parser.add_argument("--postflop-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, preflop = load_run(args.preflop_run.resolve(), args.preflop_round)
    _, postflop = load_run(args.postflop_run.resolve(), args.postflop_round)
    preflop.load_weights(str(args.preflop_weights.resolve()))
    postflop.load_weights(str(args.postflop_weights.resolve()))
    mx.eval(preflop.parameters(), postflop.parameters())
    preflop_json = scorer_json(preflop)
    postflop_json = scorer_json(postflop)
    bundle = {
        "schema": NETWORK_SCHEMA,
        "input_size": INPUT_FEATURE_COUNT,
        "strategy_transform": "softmax",
        "networks": [preflop_json, preflop_json],
        "postflop_networks": [postflop_json, postflop_json],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
