#!/usr/bin/env python3
"""Evaluate trained TIGER-style decoders with exact valid-prefix constraints.

This evaluator never creates an invalid SID tuple and never applies numeric
nearest-code mapping. It can increase the internal beam until a requested
number of distinct, history-filtered items is available, then truncates the
returned list to that requested size.
"""

import argparse
import json
import math
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import tiger_standalone as ts


def build_trie(item_codes):
    children0 = sorted({int(code[0]) for code in item_codes})
    children1 = defaultdict(set)
    children2 = defaultdict(set)
    code_items = defaultdict(list)
    for item_id, raw in enumerate(item_codes):
        code = tuple(int(x) for x in raw)
        children1[code[0]].add(code[1])
        children2[(code[0], code[1])].add(code[2])
        code_items[code].append(item_id)
    return (
        children0,
        {key: sorted(value) for key, value in children1.items()},
        {key: sorted(value) for key, value in children2.items()},
        {key: sorted(value) for key, value in code_items.items()},
    )


@torch.no_grad()
def constrained_beam(model, input_codes, trie, beam_width, device, max_code_len):
    children0, children1, children2, code_items = trie
    prefix = [ts.BOS_ID] + input_codes[-max_code_len:]
    prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)

    logits = model(prefix_tensor)[0, -1]
    logprobs = F.log_softmax(logits[:ts.CODEBOOK_K], dim=-1)
    ids0 = torch.tensor(children0, dtype=torch.long, device=device)
    scores0 = logprobs[ids0]
    keep = min(beam_width, ids0.numel())
    scores0, order = scores0.topk(keep)
    ids0 = ids0[order]

    expanded = torch.cat(
        [prefix_tensor.expand(keep, -1), ids0.unsqueeze(1)], dim=1
    )
    logits1 = F.log_softmax(
        model(expanded)[:, -1, :ts.CODEBOOK_K], dim=-1
    )
    candidate1_rows = []
    candidate1_code0 = []
    candidate1_code1 = []
    for row, code0 in enumerate(ids0.tolist()):
        for code1 in children1[code0]:
            candidate1_rows.append(row)
            candidate1_code0.append(code0)
            candidate1_code1.append(code1)
    rows1_all = torch.tensor(candidate1_rows, dtype=torch.long, device=device)
    ids1_0_all = torch.tensor(candidate1_code0, dtype=torch.long, device=device)
    ids1_1_all = torch.tensor(candidate1_code1, dtype=torch.long, device=device)
    scores1_all = scores0[rows1_all] + logits1[rows1_all, ids1_1_all]
    keep1 = min(beam_width, scores1_all.numel())
    scores1, order1 = scores1_all.topk(keep1)
    ids1_0 = ids1_0_all[order1]
    ids1_1 = ids1_1_all[order1]
    expanded2 = torch.cat(
        [
            prefix_tensor.expand(keep1, -1),
            ids1_0.unsqueeze(1),
            ids1_1.unsqueeze(1),
        ],
        dim=1,
    )
    logits2 = F.log_softmax(
        model(expanded2)[:, -1, :ts.CODEBOOK_K], dim=-1
    )
    candidate2_rows = []
    candidate2_code0 = []
    candidate2_code1 = []
    candidate2_code2 = []
    for row, (code0, code1) in enumerate(zip(ids1_0.tolist(), ids1_1.tolist())):
        for code2 in children2[(code0, code1)]:
            candidate2_rows.append(row)
            candidate2_code0.append(code0)
            candidate2_code1.append(code1)
            candidate2_code2.append(code2)
    rows2_all = torch.tensor(candidate2_rows, dtype=torch.long, device=device)
    ids2_0_all = torch.tensor(candidate2_code0, dtype=torch.long, device=device)
    ids2_1_all = torch.tensor(candidate2_code1, dtype=torch.long, device=device)
    ids2_2_all = torch.tensor(candidate2_code2, dtype=torch.long, device=device)
    scores2_all = scores1[rows2_all] + logits2[rows2_all, ids2_2_all]
    keep2 = min(beam_width, scores2_all.numel())
    scores2, order2 = scores2_all.topk(keep2)
    ids2_0 = ids2_0_all[order2].tolist()
    ids2_1 = ids2_1_all[order2].tolist()
    ids2_2 = ids2_2_all[order2].tolist()
    final_scores = scores2.tolist()

    # Collision handling is deterministic: equal-score items sharing an exact
    # code are ordered by item ID. No artificial code-distance is introduced.
    ranked = []
    for score, code0, code1, code2 in zip(
        final_scores, ids2_0, ids2_1, ids2_2
    ):
        for item_id in code_items[(code0, code1, code2)]:
            ranked.append((item_id, score))
    ranked.sort(key=lambda value: (-value[1], value[0]))
    return ranked


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    code_path = os.path.join(args.output_dir, "kmeans_codes.pt")
    item_codes = torch.load(
        code_path, map_location="cpu", weights_only=False
    )["codes"]
    with open(args.data_path, "rb") as handle:
        data = pickle.load(handle)

    max_code_len = args.max_items_per_seq * ts.NUM_LEVELS
    _, _, test_pairs = ts.build_code_sequences(
        data["train"],
        data["val"],
        data["test"],
        item_codes,
        max_items_per_seq=args.max_items_per_seq,
    )
    user_ids = [
        uid
        for uid in sorted(data["train"])
        if len(data["train"][uid]) >= 2
        and uid in data["test"]
        and uid in data["val"]
    ]
    model = ts.TransformerDecoder(
        vocab_size=ts.VOCAB_SIZE,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_len=max_code_len + ts.NUM_LEVELS + 2,
    ).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(args.output_dir, args.checkpoint),
            map_location=device,
            weights_only=False,
        )
    )
    model.eval()
    trie = build_trie(item_codes)

    cutoffs = sorted({10, 30, 50, 70, 90, 100, args.return_size})
    hits = {k: 0 for k in cutoffs}
    ndcg = {k: 0.0 for k in cutoffs}
    per_user = {}
    lengths = []
    beams_used = []
    start = time.time()

    total = len(test_pairs) if args.max_users <= 0 else min(args.max_users, len(test_pairs))
    for index, (input_codes, target, seen) in enumerate(test_pairs[:total]):
        internal_beam = args.initial_beam
        filtered = []
        while True:
            ranked = constrained_beam(
                model, input_codes, trie, internal_beam, device, max_code_len
            )
            filtered = [(item, score) for item, score in ranked if item not in seen]
            if len(filtered) >= args.return_size or internal_beam >= args.max_beam:
                break
            internal_beam = min(args.max_beam, internal_beam * 2)

        returned = filtered[: args.return_size]
        items = [item for item, _ in returned]
        lengths.append(len(items))
        beams_used.append(internal_beam)
        for cutoff in cutoffs:
            top = items[:cutoff]
            if target in top:
                hits[cutoff] += 1
                rank = top.index(target)
                ndcg[cutoff] += 1.0 / math.log2(rank + 2)
        per_user[str(user_ids[index])] = {
            "target_item": int(target),
            "top_items": items,
            "n_candidates_filtered": len(filtered),
            "returned_length": len(items),
            "internal_beam": internal_beam,
        }
        if (index + 1) % 200 == 0:
            elapsed = time.time() - start
            print(
                f"{index + 1}/{total} users; {elapsed:.1f}s; "
                f"mean length={np.mean(lengths):.2f}",
                flush=True,
            )

    metrics = {
        "protocol": "valid-prefix constrained decoding; exact-code mapping; "
        "deterministic collision tie-break; adaptive internal beam; no nearest-code fallback",
        "return_size": args.return_size,
        "initial_beam": args.initial_beam,
        "max_beam": args.max_beam,
        "num_users": total,
        "mean_returned_length": float(np.mean(lengths)),
        "min_returned_length": int(np.min(lengths)),
        "fraction_full_length": float(np.mean(np.asarray(lengths) == args.return_size)),
        "mean_internal_beam": float(np.mean(beams_used)),
        "max_internal_beam_used": int(np.max(beams_used)),
        "seconds": time.time() - start,
    }
    for cutoff in cutoffs:
        metrics[f"Recall@{cutoff}"] = hits[cutoff] / total
        metrics[f"NDCG@{cutoff}"] = ndcg[cutoff] / total

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as handle:
        json.dump({"metrics": metrics, "per_user": per_user}, handle)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default="decoder_best.pt")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--return_size", type=int, default=200)
    parser.add_argument("--initial_beam", type=int, default=200)
    parser.add_argument("--max_beam", type=int, default=1600)
    parser.add_argument("--max_users", type=int, default=0)
    parser.add_argument("--max_items_per_seq", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
