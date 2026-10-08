"""
TIGER: Generative recommendation via discrete semantic IDs.

Implements the full TIGER paradigm end-to-end:
  Phase 1: Hierarchical k-means to assign 3-level semantic IDs to items
  Phase 2: Convert user sequences to code sequences for training
  Phase 3: Train a causal Transformer decoder to predict next-item codes
  Phase 4: Beam search decoding + evaluation (Recall@K, NDCG@K)

Inputs:
  --data_path   : ml1m_processed.pkl (leave-one-out split)
  --emb_path    : item_embeddings.pt (num_items × emb_dim SASRec embeddings)

Outputs (in --output_dir):
  kmeans_codes.pt  : item semantic IDs + centroids
  decoder_best.pt  : best decoder checkpoint by val Recall@10
  results.json     : final test metrics

Dependencies: torch, numpy, sklearn (all pre-installed on target server).
"""

import argparse
import json
import logging
import math
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
VOCAB_SIZE = 259  # 0-255 codebook + PAD + BOS + EOS
CODEBOOK_K = 256
NUM_LEVELS = 3


# ---------------------------------------------------------------------------
# Phase 1: Hierarchical K-Means
# ---------------------------------------------------------------------------

def hierarchical_kmeans(embeddings: np.ndarray, n_levels: int = 3, k: int = 256,
                        seed: int = 42):
    """Assign each item a 3-level semantic ID via residual k-means.

    Returns:
        codes: (num_items, n_levels) int array of cluster assignments
        centroids: list of (k, dim) arrays per level
    """
    n_items, dim = embeddings.shape
    codes = np.zeros((n_items, n_levels), dtype=np.int32)
    centroids = []
    residual = embeddings.copy()

    for level in range(n_levels):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        km.fit(residual)
        codes[:, level] = km.labels_
        centroids.append(km.cluster_centers_.copy())
        residual = residual - km.cluster_centers_[km.labels_]
        inertia = np.sum(residual ** 2)
        log.info(f"  Level {level}: inertia={inertia:.4f}")

    unique_codes = set(map(tuple, codes.tolist()))
    uniqueness = len(unique_codes) / n_items
    log.info(f"  Unique codes: {len(unique_codes)}/{n_items} ({uniqueness:.4%})")
    log.info(f"[REACHABLE] {uniqueness:.4f} ({len(unique_codes)}/{n_items} unique codes)")
    return codes, centroids


# ---------------------------------------------------------------------------
# Phase 2: Data Preparation
# ---------------------------------------------------------------------------

def build_code_sequences(train_dict, val_dict, test_dict, item_codes,
                         max_items_per_seq: int = 50):
    """Convert user item sequences to flattened code sequences.

    Returns train_seqs, val_pairs, test_pairs where:
      train_seqs: list of (input_codes, target_codes) for training
      val_pairs:  list of (input_codes, target_item_id) for validation
      test_pairs: list of (input_codes, target_item_id, seen_items) for test
    """
    train_seqs = []
    val_pairs = []
    test_pairs = []

    for uid in sorted(train_dict.keys()):
        items = train_dict[uid]
        if len(items) < 2:
            continue

        # Training pairs: for each prefix ending at position t, predict item at t+1
        for t in range(1, len(items)):
            hist = items[max(0, t - max_items_per_seq):t]
            target_item = items[t]
            input_codes = []
            for item_id in hist:
                input_codes.extend(item_codes[item_id].tolist())
            target_codes = item_codes[target_item].tolist()
            train_seqs.append((input_codes, target_codes))

        # Validation: full train sequence → predict val item
        if uid in val_dict:
            hist = items[-max_items_per_seq:]
            input_codes = []
            for item_id in hist:
                input_codes.extend(item_codes[item_id].tolist())
            val_pairs.append((input_codes, val_dict[uid]))

        # Test: train + val → predict test item
        if uid in test_dict and uid in val_dict:
            hist_items = items + [val_dict[uid]]
            hist_items = hist_items[-max_items_per_seq:]
            input_codes = []
            for item_id in hist_items:
                input_codes.extend(item_codes[item_id].tolist())
            seen = set(items)
            if uid in val_dict:
                seen.add(val_dict[uid])
            test_pairs.append((input_codes, test_dict[uid], seen))

    log.info(f"  Train pairs: {len(train_seqs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    return train_seqs, val_pairs, test_pairs


class CodeSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, max_code_len: int = 150):
        self.pairs = pairs
        self.max_code_len = max_code_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_codes, target_codes = self.pairs[idx]
        # Truncate input to max_code_len
        input_codes = input_codes[-self.max_code_len:]
        # Full sequence: BOS + input_codes + target_codes
        full_seq = [BOS_ID] + input_codes + target_codes
        return full_seq

    @staticmethod
    def collate_fn(batch):
        max_len = max(len(seq) for seq in batch)
        padded = []
        for seq in batch:
            padded.append(seq + [PAD_ID] * (max_len - len(seq)))
        return torch.tensor(padded, dtype=torch.long)


# ---------------------------------------------------------------------------
# Phase 3: Transformer Decoder
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    """Causal Transformer decoder for next-item code prediction.

    Predicts the 3-token semantic ID of the next item given the code sequence
    of the user's interaction history. Uses learned positional embeddings and
    pre-norm Transformer blocks with causal masking.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=384, nhead=6,
                 num_layers=4, d_ff=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.drop(self.token_emb(x) + self.pos_emb(pos))

        if mask is None:
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )

        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        return self.head(h)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + h
        x = x + self.ff(self.ff_norm(x))
        return x


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_decoder(model, train_seqs, val_pairs, item_codes, device,
                  iterations=20000, batch_size=64, lr=1e-3, weight_decay=0.01,
                  warmup_steps=1000, eval_every=2000, max_code_len=150,
                  output_dir="tiger_output", beam_width=20):

    dataset = CodeSequenceDataset(train_seqs, max_code_len=max_code_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=CodeSequenceDataset.collate_fn, num_workers=4,
        pin_memory=True, drop_last=True,
    )
    loader_iter = iter(loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 0.5 * (1 + math.cos(
            math.pi * (step - warmup_steps) / (iterations - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    best_recall10 = 0.0
    model.train()
    running_loss = 0.0
    log_interval = 500

    for step in range(1, iterations + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = batch.to(device)
        # Input: all tokens except last; Target: all tokens except first
        inp = batch[:, :-1]
        tgt = batch[:, 1:]

        logits = model(inp)

        # Only compute loss on the last 3 positions (next item's codes)
        # The target codes are the last 3 tokens in each sequence (before padding)
        # Find the last non-pad position for each sequence
        pad_mask = (tgt != PAD_ID)
        seq_lens = pad_mask.sum(dim=1)  # (B,)

        loss = 0.0
        count = 0
        for b in range(inp.size(0)):
            end = seq_lens[b].item()
            if end < 3:
                continue
            for offset in range(3):
                pos = end - 3 + offset
                loss = loss + F.cross_entropy(logits[b, pos], tgt[b, pos])
                count += 1

        if count > 0:
            loss = loss / count

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            cur_lr = scheduler.get_last_lr()[0]
            log.info(f"  Step {step}/{iterations}  loss={avg_loss:.4f}  lr={cur_lr:.6f}")
            running_loss = 0.0

        if step % eval_every == 0:
            model.eval()
            metrics = evaluate(model, val_pairs, item_codes, device,
                               max_code_len=max_code_len, beam_width=beam_width,
                               max_users=500)
            r10 = metrics["Recall@10"]
            log.info(f"  [Val step={step}] R@10={r10:.4f} R@30={metrics['Recall@30']:.4f} "
                     f"R@50={metrics['Recall@50']:.4f} R@70={metrics['Recall@70']:.4f} "
                     f"R@90={metrics['Recall@90']:.4f} N@10={metrics['NDCG@10']:.4f}")
            torch.save(model.state_dict(),
                       os.path.join(output_dir, f"decoder_step{step}.pt"))
            log.info(f"  Saved checkpoint decoder_step{step}.pt")
            if r10 > best_recall10:
                best_recall10 = r10
                torch.save(model.state_dict(),
                           os.path.join(output_dir, "decoder_best.pt"))
                log.info(f"  Saved best model (R@10={r10:.4f})")
            model.train()

    return best_recall10


# ---------------------------------------------------------------------------
# Phase 4: Beam Search + Evaluation
# ---------------------------------------------------------------------------

def beam_search_predict(model, input_codes, item_codes_tensor, beam_width,
                        device, max_code_len=150):
    """Decode only catalog-valid SID prefixes and resolve exact codes.

    Cluster identifiers are categorical, so numeric distance between code
    tuples is not meaningful. Invalid prefixes are therefore masked at every
    level and completed tuples are resolved only through exact catalog lookup.
    Items sharing a tuple are ordered deterministically by item ID.
    """
    model.eval()
    codes = item_codes_tensor.detach().cpu().tolist()
    children0 = sorted({int(code[0]) for code in codes})
    children1 = defaultdict(set)
    children2 = defaultdict(set)
    code_items = defaultdict(list)
    for item_id, raw_code in enumerate(codes):
        code = tuple(int(value) for value in raw_code)
        children1[code[0]].add(code[1])
        children2[(code[0], code[1])].add(code[2])
        code_items[code].append(item_id)

    prefix = [BOS_ID] + input_codes[-max_code_len:]
    prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)

    with torch.no_grad():
        logits0 = F.log_softmax(
            model(prefix_tensor)[0, -1, :CODEBOOK_K], dim=-1
        )
    ids0_all = torch.tensor(children0, dtype=torch.long, device=device)
    keep0 = min(beam_width, ids0_all.numel())
    scores0, order0 = logits0[ids0_all].topk(keep0)
    ids0 = ids0_all[order0]

    expanded1 = torch.cat(
        [prefix_tensor.expand(keep0, -1), ids0.unsqueeze(1)], dim=1
    )
    with torch.no_grad():
        logits1 = F.log_softmax(
            model(expanded1)[:, -1, :CODEBOOK_K], dim=-1
        )
    rows1, codes10, codes11 = [], [], []
    for row, code0 in enumerate(ids0.tolist()):
        for code1 in sorted(children1[code0]):
            rows1.append(row)
            codes10.append(code0)
            codes11.append(code1)
    rows1 = torch.tensor(rows1, dtype=torch.long, device=device)
    ids10 = torch.tensor(codes10, dtype=torch.long, device=device)
    ids11 = torch.tensor(codes11, dtype=torch.long, device=device)
    scores1_all = scores0[rows1] + logits1[rows1, ids11]
    keep1 = min(beam_width, scores1_all.numel())
    scores1, order1 = scores1_all.topk(keep1)
    ids10, ids11 = ids10[order1], ids11[order1]

    expanded2 = torch.cat(
        [
            prefix_tensor.expand(keep1, -1),
            ids10.unsqueeze(1),
            ids11.unsqueeze(1),
        ],
        dim=1,
    )
    with torch.no_grad():
        logits2 = F.log_softmax(
            model(expanded2)[:, -1, :CODEBOOK_K], dim=-1
        )
    rows2, codes20, codes21, codes22 = [], [], [], []
    for row, (code0, code1) in enumerate(zip(ids10.tolist(), ids11.tolist())):
        for code2 in sorted(children2[(code0, code1)]):
            rows2.append(row)
            codes20.append(code0)
            codes21.append(code1)
            codes22.append(code2)
    rows2 = torch.tensor(rows2, dtype=torch.long, device=device)
    ids20 = torch.tensor(codes20, dtype=torch.long, device=device)
    ids21 = torch.tensor(codes21, dtype=torch.long, device=device)
    ids22 = torch.tensor(codes22, dtype=torch.long, device=device)
    scores2_all = scores1[rows2] + logits2[rows2, ids22]
    keep2 = min(beam_width, scores2_all.numel())
    scores2, order2 = scores2_all.topk(keep2)

    ranked = []
    for score, code0, code1, code2 in zip(
        scores2.tolist(),
        ids20[order2].tolist(),
        ids21[order2].tolist(),
        ids22[order2].tolist(),
    ):
        for item_id in sorted(code_items[(code0, code1, code2)]):
            ranked.append((item_id, score))
    ranked.sort(key=lambda value: (-value[1], value[0]))
    return ranked


def evaluate(model, eval_pairs, item_codes, device, max_code_len=150,
             beam_width=100, max_users=None, per_user_save_path=None, user_ids=None):
    """Evaluate the model on val/test pairs.

    Returns dict with Recall@K and NDCG@K for K in {10, 30, 50, 70, 90}.
    """
    model.eval()
    item_codes_tensor = torch.tensor(item_codes, dtype=torch.long, device=device)

    ks = [10, 30, 50, 70, 90]
    recalls = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    reachable = 0
    total = 0

    pairs = eval_pairs
    indices = None
    if max_users is not None and max_users < len(pairs):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(pairs), max_users, replace=False)
        pairs = [eval_pairs[i] for i in indices]

    if user_ids is not None:
        sampled_user_ids = [user_ids[i] for i in indices] if indices is not None else list(user_ids)
    else:
        sampled_user_ids = None
    per_user_results = {} if per_user_save_path else None

    for idx, pair in enumerate(pairs):
        if len(pair) == 3:
            input_codes, target_item, seen_items = pair
        else:
            input_codes, target_item = pair
            seen_items = set()

        internal_beam = beam_width
        while True:
            ranked = beam_search_predict(
                model, input_codes, item_codes_tensor,
                internal_beam, device, max_code_len
            )
            filtered = [(iid, score) for iid, score in ranked
                        if iid not in seen_items]
            if len(filtered) >= beam_width or internal_beam >= 1600:
                break
            internal_beam = min(1600, internal_beam * 2)
        filtered = filtered[:beam_width]

        # Check reachability
        target_in_beam = any(iid == target_item for iid, _ in filtered)
        if target_in_beam:
            reachable += 1

        for k in ks:
            top_k = [iid for iid, _ in filtered[:k]]
            if target_item in top_k:
                recalls[k] += 1.0
                rank = top_k.index(target_item)
                ndcgs[k] += 1.0 / math.log2(rank + 2)

        total += 1

        if per_user_results is not None:
            uid_key = str(sampled_user_ids[idx]) if sampled_user_ids else str(idx)
            user_data = {
                'target_item': int(target_item),
                'reachable': target_in_beam,
                'n_candidates_raw': len(ranked),
                'n_candidates_filtered': len(filtered),
                'n_unique_predicted': len(set(iid for iid, _ in ranked)),
                'internal_beam': internal_beam,
                'raw_beam_items': [int(iid) for iid, _ in ranked],
                'top_items': [int(iid) for iid, _ in filtered[:beam_width]],
            }
            for k in ks:
                top_k = [iid for iid, _ in filtered[:k]]
                hit = target_item in top_k
                user_data[f'R@{k}'] = 1.0 if hit else 0.0
                if hit:
                    rk = top_k.index(target_item)
                    user_data[f'NDCG@{k}'] = 1.0 / math.log2(rk + 2)
                else:
                    user_data[f'NDCG@{k}'] = 0.0
            per_user_results[uid_key] = user_data

        if (idx + 1) % 200 == 0:
            log.info(f"    Eval progress: {idx+1}/{len(pairs)}")

    metrics = {}
    for k in ks:
        metrics[f"Recall@{k}"] = recalls[k] / total if total > 0 else 0.0
        metrics[f"NDCG@{k}"] = ndcgs[k] / total if total > 0 else 0.0
    metrics["Reachable_Rate"] = reachable / total if total > 0 else 0.0
    metrics["Num_Users"] = total

    if per_user_save_path and per_user_results:
        with open(per_user_save_path, 'w') as f:
            json.dump(per_user_results, f)
        log.info(f"Saved per-user predictions ({len(per_user_results)} users) to {per_user_save_path}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TIGER generative recommendation")
    parser.add_argument("--data_path", type=str, default="data/ml1m_processed.pkl")
    parser.add_argument("--emb_path", type=str, default="checkpoints/item_embeddings.pt")
    parser.add_argument("--output_dir", type=str, default="tiger_output")
    parser.add_argument("--max_items_per_seq", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--beam_width", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--eval_beam_width", type=int, default=20,
                        help="Beam width for validation during training (smaller=faster)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ---- Phase 1: Hierarchical K-Means ----
    log.info("Phase 1: Hierarchical K-Means")
    emb_data = torch.load(args.emb_path, map_location="cpu")
    if isinstance(emb_data, dict):
        embeddings = emb_data.get("embeddings", emb_data.get("item_embeddings"))
    else:
        embeddings = emb_data
    embeddings = embeddings.numpy().astype(np.float32)
    log.info(f"  Loaded embeddings: {embeddings.shape}")

    codes, centroids = hierarchical_kmeans(embeddings, n_levels=NUM_LEVELS,
                                           k=CODEBOOK_K, seed=args.seed)
    torch.save({"codes": codes, "centroids": centroids},
               os.path.join(args.output_dir, "kmeans_codes.pt"))
    log.info("  Saved kmeans_codes.pt")

    # ---- Phase 2: Data Preparation ----
    log.info("Phase 2: Data Preparation")
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    train_dict = data["train"]
    val_dict = data["val"]
    test_dict = data["test"]
    log.info(f"  Users: train={len(train_dict)}, val={len(val_dict)}, test={len(test_dict)}")

    max_code_len = args.max_items_per_seq * NUM_LEVELS
    train_seqs, val_pairs, test_pairs = build_code_sequences(
        train_dict, val_dict, test_dict, codes,
        max_items_per_seq=args.max_items_per_seq,
    )

    # ---- Phase 3: Training ----
    log.info("Phase 3: Decoder Training")
    model = TransformerDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_len=max_code_len + NUM_LEVELS + 2,  # +BOS +margin
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Model params: {num_params:,}")

    best_r10 = train_decoder(
        model, train_seqs, val_pairs, codes, device,
        iterations=args.iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=args.eval_every,
        max_code_len=max_code_len,
        output_dir=args.output_dir,
        beam_width=args.eval_beam_width,
    )

    # ---- Phase 4: Final Evaluation ----
    log.info("Phase 4: Final Test Evaluation")
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "decoder_best.pt"),
                    map_location=device)
    )
    model.eval()

    test_user_ids = [uid for uid in sorted(train_dict.keys())
                     if len(train_dict[uid]) >= 2 and uid in test_dict and uid in val_dict]

    log.info(f"Final eval capped at 5000/{len(test_pairs)} users")
    test_metrics = evaluate(model, test_pairs, codes, device,
                            max_code_len=max_code_len,
                            beam_width=args.beam_width,
                            max_users=5000,
                            per_user_save_path=os.path.join(args.output_dir, "per_user_predictions.json"),
                            user_ids=test_user_ids)

    log.info("=" * 60)
    log.info("TIGER Final Test Results:")
    for k, v in sorted(test_metrics.items()):
        log.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    log.info("=" * 60)

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    log.info(f"Results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
