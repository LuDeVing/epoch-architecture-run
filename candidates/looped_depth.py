# IDEA: Second pass over block subset (looped depth reuse)
# HYPOTHESIS: Reusing the upper six blocks with separate gains adds useful sequential computation without duplicating matrices.
# CHANGE: Block adds alternate n1/n2 only for the upper six blocks; GPT.blocks construction and forward route the second pass through them.
# FIDELITY: partial  Exactly repeat blocks 6-11 with separate gains; actual added count is 4,608 rather than the sketch estimate about 9k; graph is unconditional.
# PREDICTION: 45-step loss roughly baseline; kill above 6.66 or step above 1.9s; step about 1.37s, peak +0.35GB, final loss -0.01 to -0.03.
"""nano trainer: a small, readable language-model training script that the HTSA loop proposes
changes against. Fixed token budget, fixed data, fixed evaluation; everything else is fair game.

Baseline architecture (research_notes/architectures/nano_specs.md, section A, "nano-GPT2"):
pre-norm Transformer, RMSNorm, rotary positions, SwiGLU MLP, no biases, tied embeddings,
d=384, 12 layers, 6 heads x 64. Optimizer: Muon for 2-D weights, AdamW for the rest.

Modes (the verifier in nano/check.py uses them as rungs):
  --build-only            construct the model, print parameter counts, exit
  --smoke-steps N         run N optimizer steps on real data, report step time and peak memory
  --smoke-seconds S       train for S seconds, report whether loss fell
  (default)               train for --tokens tokens, evaluate, write metrics.json

Outputs metrics.json in --out: {val_loss, val_bpb, params_total, params_nonembed, tokens,
steps, wall_s, peak_vram_gb, loss_first, loss_last, loss_curve, arch, seed} and, after a full run,
model.pt (weights + config) for the harness's own exam (nano/exam.py). The data folder comes from
HTSA_DATA_DIR; under the harness it holds the train shards only, so val_loss is None there.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = Path(os.environ.get("HTSA_DATA_DIR") or (Path(os.environ.get("HTSA_LAB_ROOT", r"D:\AGI-lab")) / "data" / "fineweb"))


# ======================================================================================
# Config
# ======================================================================================
@dataclass
class Config:
    arch: str = "gpt2"
    vocab_size: int = 50304          # GPT-2 BPE (50257) padded to a multiple of 64
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6               # = n_head for MHA; fewer for GQA
    d_model: int = 384
    d_ff: int = 1024                 # SwiGLU hidden size
    seq_len: int = 1024
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    sliding_window: int = 0          # 0 = full causal attention
    # training
    tokens: float = 50e6
    batch_tokens: int = 32768        # tokens per optimizer step (micro-batches accumulate)
    micro_batch: int = 4             # sequences per micro-batch (4 x 1024 = 4096 tokens); 8 spilled past 6 GB
    lr: float = 0.02                 # Muon learning rate (matrix params)
    lr_adam: float = 3e-3            # AdamW learning rate (embeddings, norms)
    weight_decay: float = 0.1
    warmup_frac: float = 0.02
    cooldown_frac: float = 0.30
    grad_clip: float = 1.0
    lr_mult_both: float = 1.0        # scales both learning rates (the tuned-baseline duel uses it)
    seed: int = 0
    eval_tokens: int = 2_000_000
    dtype: str = "bfloat16"


# ======================================================================================
# Model
# ======================================================================================
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


def rope_cache(seq_len: int, head_dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                       # (T, D/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); rotate pairs (x1, x2) = (x[..., :D/2], x[..., D/2:])
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos, sin = cos[: x.size(2)].to(x.dtype), sin[: x.size(2)].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head, self.n_kv = cfg.n_head, cfg.n_kv_head
        self.hd = cfg.d_model // cfg.n_head
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * self.hd, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.hd, cfg.d_model, bias=False)
        self.window = cfg.sliding_window

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.n_kv != self.n_head:
            rep = self.n_head // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        if self.window and self.window < T:
            i = torch.arange(T, device=x.device)
            mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < self.window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    """SwiGLU feed-forward."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config, second_pass: bool = False):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn, self.mlp = Attention(cfg), MLP(cfg)
        if second_pass:
            self.n1_second, self.n2_second = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)

    def forward(self, x, cos, sin, second_pass: bool = False):
        n1 = self.n1_second if second_pass else self.n1
        n2 = self.n2_second if second_pass else self.n2
        x = x + self.attn(n1(x), cos, sin)
        return x + self.mlp(n2(x))


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, second_pass=(i >= cfg.n_layer // 2)) for i in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight
        self.apply(self._init)
        for b in self.blocks:                              # scaled residual-branch init
            nn.init.normal_(b.attn.wo.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))
            nn.init.normal_(b.mlp.w_down.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        cos, sin = rope_cache(T, self.cfg.d_model // self.cfg.n_head, self.cfg.rope_theta, idx.device)
        x = self.tok(idx)
        for b in self.blocks:
            x = b(x, cos, sin)
        for b in self.blocks[self.cfg.n_layer // 2:]:
            x = b(x, cos, sin, second_pass=True)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        # logits stay in the autocast dtype; a float32 copy of (B*T, 50304) alone is 0.8 GB at B=4
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def param_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    embed = model.tok.weight.numel()
    return total, total - embed


# ======================================================================================
# Muon optimizer (Newton-Schulz orthogonalized momentum; Jordan et al. 2024)
# ======================================================================================
def newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B_ = b * A + c * A @ A
        X = a * X + B_ @ X
    return (X.T if transposed else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                u = newton_schulz5(g.reshape(g.size(0), -1)).view_as(g)
                scale = max(1.0, g.size(0) / g.size(1)) ** 0.5     # shape-aware scaling
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u, alpha=-group["lr"] * scale)


def build_optimizers(model: GPT, cfg: Config):
    matrix = [p for n, p in model.named_parameters() if p.ndim == 2 and "tok" not in n and "head" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim == 2 and "tok" not in n and "head" not in n)]
    return [Muon(matrix, lr=cfg.lr, weight_decay=cfg.weight_decay),
            torch.optim.AdamW(other, lr=cfg.lr_adam, betas=(0.9, 0.95), weight_decay=cfg.weight_decay, eps=1e-10)]


def lr_mult(step: int, total: int, cfg: Config) -> float:
    warm = max(1, int(total * cfg.warmup_frac))
    cool_start = int(total * (1 - cfg.cooldown_frac))
    if step < warm:
        return (step + 1) / warm
    if step >= cool_start:
        return max(0.0, (total - step) / max(1, total - cool_start))
    return 1.0


# ======================================================================================
# Data
# ======================================================================================
class Shards:
    def __init__(self, split: str, seq_len: int, seed: int):
        files = sorted(DATA_DIR.glob("train_*.bin")) if split == "train" else [DATA_DIR / "val.bin"]
        files = [f for f in files if f.exists()]
        if not files and split == "train":
            raise FileNotFoundError(f"no train shards in {DATA_DIR}; run nano/prepare_data.py")
        # the validation split is normally absent from a training process: the harness runs the exam itself
        self.data = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.pos = 0

    def sample(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Random windows (train) from a random shard."""
        d = self.data[self.rng.integers(len(self.data))]
        starts = self.rng.integers(0, len(d) - self.seq_len - 1, size=n)
        x = np.stack([d[s: s + self.seq_len] for s in starts]).astype(np.int64)
        y = np.stack([d[s + 1: s + self.seq_len + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(device, non_blocking=True), torch.from_numpy(y).to(device, non_blocking=True)

    def sequential(self, n: int, device):
        """Fixed sequential windows (val) so every run evaluates the same tokens."""
        if not self.data:
            return None
        d = self.data[0]
        T = self.seq_len
        out_x, out_y = [], []
        for _ in range(n):
            if self.pos + T + 1 > len(d):
                return None
            out_x.append(d[self.pos: self.pos + T].astype(np.int64))
            out_y.append(d[self.pos + 1: self.pos + T + 1].astype(np.int64))
            self.pos += T
        return (torch.from_numpy(np.stack(out_x)).to(device), torch.from_numpy(np.stack(out_y)).to(device))


@torch.no_grad()
def evaluate(model: GPT, cfg: Config, device, dtype) -> float:
    model.eval()
    val = Shards("val", cfg.seq_len, cfg.seed)
    n_batches = max(1, cfg.eval_tokens // (cfg.micro_batch * cfg.seq_len))
    losses = []
    for _ in range(n_batches):
        batch = val.sequential(cfg.micro_batch, device)
        if batch is None:
            break
        x, y = batch
        with torch.autocast("cuda", dtype=dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# ======================================================================================
# Train
# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    for k, v in asdict(Config()).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v) if not isinstance(v, bool) else int, default=v)
    ap.add_argument("--out", default=None)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=0)
    ap.add_argument("--smoke-seconds", type=float, default=0.0)
    args = ap.parse_args()
    cfg = Config(**{k: getattr(args, k) for k in asdict(Config())})
    cfg.tie_embeddings = bool(cfg.tie_embeddings)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    model = GPT(cfg).to(device)
    total, nonembed = param_counts(model)
    report = {"arch": cfg.arch, "seed": cfg.seed, "params_total": total, "params_nonembed": nonembed,
              "config": asdict(cfg)}
    print(f"[nano] params total {total/1e6:.2f}M, non-embedding {nonembed/1e6:.2f}M, device {device}", flush=True)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    def write(extra: dict) -> None:
        report.update(extra)
        if out:
            (out / "metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    if args.build_only:
        write({"mode": "build"})
        print(json.dumps({k: report[k] for k in ("params_total", "params_nonembed")}))
        return 0

    opts = build_optimizers(model, cfg)
    train = Shards("train", cfg.seq_len, cfg.seed)
    accum = max(1, cfg.batch_tokens // (cfg.micro_batch * cfg.seq_len))
    total_steps = max(1, int(cfg.tokens // cfg.batch_tokens))
    if args.smoke_steps:
        total_steps = args.smoke_steps
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    t0 = time.time()
    losses: list[float] = []
    step = 0
    model.train()
    while step < total_steps:
        m = lr_mult(step, total_steps, cfg)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = (cfg.lr if isinstance(o, Muon) else cfg.lr_adam) * m * cfg.lr_mult_both
        loss_acc = 0.0
        for _ in range(accum):
            x, y = train.sample(cfg.micro_batch, device)
            with torch.autocast("cuda", dtype=dtype):
                _, loss = model(x, y)
            (loss / accum).backward()
            loss_acc += loss.item() / accum
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for o in opts:
            o.step()
        model.zero_grad(set_to_none=True)
        losses.append(loss_acc)
        step += 1
        if step % 50 == 0 or step == 1:
            el = time.time() - t0
            print(f"[nano] step {step}/{total_steps} loss {loss_acc:.4f} lr x{m:.2f} "
                  f"{step * cfg.batch_tokens / el / 1e3:.1f}k tok/s", flush=True)
        if args.smoke_seconds and time.time() - t0 > args.smoke_seconds:
            break
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    res = {"steps": step, "tokens": step * cfg.batch_tokens, "wall_s": round(wall, 1),
           "peak_vram_gb": round(peak, 2), "tok_per_s": round(step * cfg.batch_tokens / max(wall, 1e-6)),
           "loss_first": round(float(np.mean(losses[:5])), 4) if losses else None,
           "loss_last": round(float(np.mean(losses[-5:])), 4) if losses else None,
           "loss_curve": [round(float(l), 4) for l in losses]}   # per-step curve: matched-step comparisons
    if args.smoke_steps or args.smoke_seconds:
        res["mode"] = "smoke"
        write(res)
        print(json.dumps(res))
        return 0
    if out:                                   # the harness's exam (nano/exam.py) reads these weights
        torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, out / "model.pt")
    val_loss = evaluate(model, cfg, device, dtype)   # a self-report; absent when the split is hidden
    has_val = val_loss == val_loss
    res.update({"mode": "full", "val_loss": round(val_loss, 4) if has_val else None,
                "val_bpb": round(val_loss / math.log(2) * 0.2446, 4) if has_val else None})   # tokens/byte ~0.2446 for GPT-2 BPE on English web text (approx)
    write(res)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
