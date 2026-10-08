#!/usr/bin/env python3
"""Reselect a TIGER checkpoint with valid-prefix constrained validation."""

import argparse
import json
import os
import pickle
import shutil
import time

import numpy as np
import torch

import tiger_standalone as ts
from tiger_constrained_eval import build_trie, constrained_beam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--validation_users", type=int, default=500)
    parser.add_argument("--return_size", type=int, default=100)
    parser.add_argument("--initial_beam", type=int, default=100)
    parser.add_argument("--max_beam", type=int, default=800)
    parser.add_argument("--max_items_per_seq", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item_codes = torch.load(
        os.path.join(args.output_dir, "kmeans_codes.pt"),
        map_location="cpu",
        weights_only=False,
    )["codes"]
    with open(args.data_path, "rb") as handle:
        data = pickle.load(handle)
    max_code_len = args.max_items_per_seq * ts.NUM_LEVELS
    _, val_pairs, _ = ts.build_code_sequences(
        data["train"], data["val"], data["test"], item_codes,
        max_items_per_seq=args.max_items_per_seq,
    )
    rng = np.random.RandomState(42)
    indices = rng.choice(
        len(val_pairs), min(args.validation_users, len(val_pairs)), replace=False
    )
    pairs = [val_pairs[index] for index in indices]
    trie = build_trie(item_codes)

    model = ts.TransformerDecoder(
        vocab_size=ts.VOCAB_SIZE,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_len=max_code_len + ts.NUM_LEVELS + 2,
    ).to(device)
    checkpoints = sorted(
        name
        for name in os.listdir(args.output_dir)
        if name.startswith("decoder_step") and name.endswith(".pt")
    )
    results = {}
    for checkpoint in checkpoints:
        model.load_state_dict(
            torch.load(
                os.path.join(args.output_dir, checkpoint),
                map_location=device,
                weights_only=False,
            )
        )
        model.eval()
        hits = 0
        lengths = []
        beams = []
        start = time.time()
        for pair in pairs:
            if len(pair) == 3:
                input_codes, target, seen = pair
            else:
                input_codes, target = pair
                seen = set()
            internal_beam = args.initial_beam
            while True:
                ranked = constrained_beam(
                    model, input_codes, trie, internal_beam, device, max_code_len
                )
                filtered = [
                    (item, score) for item, score in ranked if item not in seen
                ]
                if (
                    len(filtered) >= args.return_size
                    or internal_beam >= args.max_beam
                ):
                    break
                internal_beam = min(args.max_beam, internal_beam * 2)
            items = [item for item, _ in filtered[: args.return_size]]
            hits += int(target in items[:10])
            lengths.append(len(items))
            beams.append(internal_beam)
        results[checkpoint] = {
            "Recall@10": hits / len(pairs),
            "mean_returned_length": float(np.mean(lengths)),
            "min_returned_length": int(np.min(lengths)),
            "mean_internal_beam": float(np.mean(beams)),
            "seconds": time.time() - start,
        }
        print(checkpoint, results[checkpoint], flush=True)

    # Deterministic tie break favors the earlier training step.
    best = max(
        checkpoints,
        key=lambda name: (
            results[name]["Recall@10"],
            -int(name.removeprefix("decoder_step").removesuffix(".pt")),
        ),
    )
    output = {
        "protocol": "valid-prefix constrained validation with exact-code mapping",
        "validation_users": len(pairs),
        "selection_metric": "Recall@10",
        "best_checkpoint": best,
        "checkpoints": results,
    }
    with open(args.out_json, "w") as handle:
        json.dump(output, handle, indent=2)
    shutil.copyfile(
        os.path.join(args.output_dir, best),
        os.path.join(args.output_dir, "decoder_best_constrained.pt"),
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
