# Build briefs for the hand-built step

Run skel-1__nano__gpt2_50m__scaffold__r0. 275 sketches, 139 clusters. The 19 clusters below are the build slots the loop had assigned (by score and from the long-shot lane, final assignment after interrogation); after them come the eight interrogated ideas with everything the questioning produced. Everything a builder would have seen is here except the training script itself and the cost card.

## Per-token scale and bias decoupling norm from direction  (picked_main; part Inputs and outputs: tied embedding, stream read-out, logits; inward; prior: new)
**Direction-only tie with per-token scale and bias** (leap, card research, p_success 0.15)
- mechanism: Keep the shared 50304x384 table but use its RMS-normalized rows as the output classifier, multiplied by a learned per-token log-scale and plus a learned per-token bias (two vectors of 50304 scalars, zero-init). The tie currently forces one row norm to serve as input salience and as output frequency prior at once, and the 2026 tying analysis shows the output gradient wins that fight; decoupling norm from direction lets the shared direction serve both roles while rare rows (rank 10k+, 11.6 nats) keep a usable direction even when decay shrinks their norm. The conflict is intrinsic to tying and do
- prediction: Loss curve pulls below baseline within the first few hundred steps as the bias absorbs the unigram prior, and the gap persists; parameters rise by about 100k to roughly 21.34M; step time within 2 percent (one 50304x384 normalize per forward, done in bf16 to stay under the 0.1 GB memory headroom); th
- test: Build-only for the count, 3 steps for time and peak memory, then 45 matched steps against baseline and against the bias-only variant below to isolate the normalization.
- breaks: The tied row's norm must encode both input salience and output frequency; this flips it so norm is output-only and lives in two cheap scalar vectors.
- keeps: One shared table, plain next-token cross entropy, final RMSNorm, mean read-out, the same Muon and AdamW grouping (the two new vectors fall to AdamW as 1-D parameters).
- kill: Mean training loss over matched steps 30 to 45 not at least 0.02 nats below baseline, or below the bias-only variant by less than 0.005.
- target: loss on rare tokens, down
- scope: validation loss at 1024
- source: own, building on https://arxiv.org/pdf/2603.26663

Other sketches in this group:
- Counted bias plus μ-centering of output directions (port): Combine the counted bias and direction-only head with μ-centering: subtract the mean normalised row after each step, so the common push-down direction the bias now carries cannot reappear in the rows.
- Direction-only head plus chunked fp32 cross-entropy (port): Combine with Cut Cross-Entropy-style chunking (https://arxiv.org/abs/2411.09009). Compute logits and loss in fp32 over chunks of 1024 tokens (about 206 MB per chunk), so the bias b and scale s act at fp32 resolution instead of the 0.0625 bf16 step near logit 10. Chunking also lowers peak VRAM compar
- Direction-only tied head with a counted unigram bias (leap): The brief's centering paper identifies output anisotropy, and the rare-token loss equals the unigram cost; literalise "row norm equals log frequency" and the consequence is that the tied table spends its norm, which the input lookup also sees, on a quantity a bias can hold. Logits become g·norm(x)·(

### Interrogation

chain:
  -
    - id: L1
    - text: Head gets a per-token bias b initialised to counted log-unigram, kept out of weight decay -> the frequency prior is in the logits from step 0 instead of being learned through row geometry
    - quantity: training loss at matched steps 1–26
    - baseline: 10.88 at step 1, 7.06 at step 26
    - expect: down 1–3 nats at steps 1–10, down 0.03–0.3 at step 26; a zero-init learned bias gives under 0.05 at step 26
    - breaks_if: counted-init gap at step 26 is under 0.02 nats
  -
    - id: L2
    - text: Bias carries the prior -> rows stop sharing a common "push-down" direction
    - quantity: cosine of each output row to the mean output row (anisotropy), and ‖mean row‖ / mean ‖row‖
    - baseline: unmeasured
    - expect: down by 20–60% relative at step 100
    - breaks_if: anisotropy at step 100 is within 10% of baseline
  -
    - id: L3
    - text: Rows normalised in the head -> the output gradient has no radial component, so row norm is set only by the input lookup and decay
    - quantity: Spearman correlation of row norm with log train frequency
    - baseline: unmeasured (belief: strongly positive)
    - expect: falls toward 0 to 0.3
    - breaks_if: correlation stays above 0.5 at step 100
  -
    - id: L4
    - text: Input rows no longer carry the output prior -> early layers get cleaner token features, so mid-frequency conditional prediction improves beyond what bias alone gives
    - quantity: loss on rank 100–1000 and 1000–10000 bands, compared with the bias-only variant
    - baseline: 6.829 / 8.979 at step 100
    - expect: down 0.01–0.1 compared with bias-only
    - breaks_if: direction-only minus bias-only is above −0.005 on both bands over 3 seeds
  -
    - id: L5
    - text: Early gain persists through 1525 steps -> exam loss falls
    - quantity: training loss at steps 1000–1525 and exam val loss
    - baseline: unmeasured here
    - expect: down 0.005–0.03 nats
    - breaks_if: the matched training-loss gap over the last 300 steps is under 0.005

weakest:
  L4

qa:
  -
    - q: With zero-init log-scale s and RMS-normalised rows, what is the initial logit scale?
    - a: RMS-normalised rows have RMS 1, which is 50x the baseline row RMS of 0.02. Logit std becomes about √384 ≈ 19.6 instead of 0.39, and the output gradient into the tied table is also amplified 50x. Zero-init therefore starts at a loss of tens of nats and makes the output side dominate even more, the opposite of the intent. Initialise s = log(0.02) so the head equals the baseline head at step 0, or equivalently L2-normalise and set s = log(0.392).
    - decides: s init = log(0.02) ≈ −3.91 vs s = 0
  -
    - q: Can a zero-init learned bias absorb the unigram prior "within the first few hundred steps" under AdamW lr 3e-3?
    - a: No. Adam moves each element by roughly the learning rate or less per step. That is at most about 0.9 nats by step 300 and about 3.5–4.5 nats over the whole schedule. The log-prior spans more than 10 nats between frequent and rank-10k+ tokens. The prediction in the idea cannot happen with zero init, so take the counted-init variant.
    - decides: zero-init learned b vs b initialised to counted log-unigram
  -
    - q: Should s and b be in the AdamW group with weight_decay = 0.1?
    - a: Decoupled decay costs lr·wd = 3e-4 per step times the value. For b ≈ −10 that is 3e-3 per step toward 0, the same size as the Adam step, and it erases roughly a third of the prior over the run. Decay on s pulls it toward exp(0) = 1, which is a 50x temperature. Put both in a no-decay AdamW group.
    - decides: the existing `other` group vs a new no-decay group
  -
    - q: How should the per-token scale and bias be applied without breaking C4?
    - a: Logits are 4096×50304 in bf16, about 412 MB. Any elementwise operation on them saves at least one more such tensor for backward, which is far beyond the roughly 0.1 GB headroom. Folding adds one 50304×384 bf16 tensor (38.6 MB) plus row statistics. The bias rides in the addmm, so no extra logits-sized tensor is created.
    - decides: fold into the weight (`W = E * (exp(s) / rms_row(E))[:, None]`, then `F.linear(h, W, b)`) vs elementwise scale/add on the logits
  -
    - q: Does normalisation beyond the bias change anything the optimiser sees?
    - a: The output gradient to a raw row becomes (exp(s)/rms)·(I − êêᵀ)g. The radial part is removed exactly, and the tangential part shrinks as rows grow above their initial norm. Under per-element Adam normalisation, magnitudes are partly washed out, so the main surviving effect is the loss of the radial push. Whether that lowers loss is unknown, and it is what the bias-only arm tests.
    - decides: RMS-normalised direction-only head plus s and b vs bias-only
  -
    - q: Is "rare-token loss down" the right target?
    - a: At step 100 the rank-10k+ band sits at 11.6 nats, already above uniform (10.83), so the model is already charging roughly a prior cost there. A correct unigram prior gives rank-10k+ tokens their unigram code length, which is plausibly 12–15 nats (belief). That can raise rare-band loss while lowering total loss. Judge on total loss and the mid bands, and only report the rare band.
    - decides: rank-10k+ band loss vs total loss plus the 100–10000 bands as the success signal
  -
    - q: Where are the counts computed?
    - a: The exam runs GPT(cfg) and then load_state_dict, so counting inside __init__ would read data at exam time and break --build-only. Count in main after the build-only return: - Source: `train.data[0]` with `np.bincount(..., minlength=50304)`. - Smoothing: add-one. The 47 padding ids then get finite, very negative values; avoid −inf in bf16. - Store with `b.copy_(log p)`. Because b is a Parameter, the exam sees it. Never touch val.bin.
    - decides: inside GPT.__init__ vs in main after the --build-only return
  -
    - q: Is the steps 30–45 kill window meaningful?
    - a: Baseline is already at 7.06 by step 26, likely at or below unigram entropy (belief: about 7.3–7.6 nats for GPT-2 BPE on FineWeb-Edu). So steps 30–45 already test whether the head start survives into the bigram phase. However, the 0.02 bar is 4x the C7 margin, and early prior gains routinely fade. Keep 30–45 as the first gate and add a 300-step matched check before any full run. Matched seeds give identical data order, because the Shards RNG is independent of the model.
    - decides: kill on steps 30–45 vs kill on a matched window at about steps 250–300
  -
    - q: Does the bias interact with global grad clipping at 1.0?
    - a: Unknown magnitude. The bias gradient is mean(p − onehot) over 32k tokens. Its norm is up to about 1 when the prior is wrong and small when it is counted. With counted init it should rarely trigger clipping. Log the pre-clip global norm in both arms.
    - decides: include s and b in clip_grad_norm_ vs clip them separately or exclude them
  -
    - q: Does the tied head's self-logit (the token scoring itself through its own embedding in the residual) get worse?
    - a: Sinkhorn mixing is doubly stochastic, so the mean over streams preserves the input exactly: the readout always contains e_x plus the branch writes. With s = log(rms0), the self-logit at init equals baseline (about 7.7). Later, as raw rows grow on the input side but not in the head, the ratio drifts in an unmeasured direction. Measure the logit of the current token in both arms.
    - decides: plain readout vs readout that subtracts tok(idx)

known_tagged:
  -
    - text: The head is `nn.Linear` with weight tied to `tok.weight`, applied to `self.norm(x.mean(dim=2))`. Logits stay bf16 under autocast and the loss is plain F.cross_entropy.
    - prov: script
  -
    - text: `param_counts` subtracts only `tok.weight`, so s and b (2 × 50304 = 100,608) count as non-embedding: 21.24M → about 21.34M, cap about 23.36M. A frozen buffer would not count.
    - prov: script
  -
    - text: `named_parameters` dedupes the tie, so only `tok.weight` appears; it goes to AdamW (lr 3e-3, wd 0.1, betas 0.9/0.95). New 1-D vectors fall to AdamW too, but need their own wd = 0 group.
    - prov: script
  -
    - text: Steps = 50e6 / 32768 = 1525; accum = 8 micro-batches of 4×1024; warmup 2%, cooldown 30%.
    - prov: script
  -
    - text: Initial row RMS is 0.02 and RMSNorm output RMS is 1. Initialise s = log(0.02) for RMS-normalised rows so step-0 logits match baseline.
    - prov: script
  -
    - text: Logits tensor per micro-batch is 4096×50304 bf16 ≈ 412 MB. Never add a logits-sized elementwise op; fold s into the weight and b into F.linear.
    - prov: script
  -
    - text: The exam builds GPT(cfg) and loads `state_dict`. The prior must be a Parameter or persistent buffer filled in main before training, never computed in `__init__`.
    - prov: script
  -
    - text: Mean over Sinkhorn-mixed streams preserves the input, so the readout always contains exactly e_x plus branch writes.
    - prov: script
  -
    - text: At step 100, the rank-10k+ band is 11.613 nats on 12% of tokens; the rank 0–100 band is 3.163 on 42%; step-26 loss is 7.06.
    - prov: measured
  -
    - text: Layer-0 gradient is anomalously high (attn 0.24–0.29, about 10x layer 1) on 3 seeds. It is a possible readout of input-side pressure; watch it in both arms.
    - prov: measured
  -
    - text: Tied matrices end up output-shaped: mean cosine 0.72 to the untied output matrix vs 0.53 to the input matrix (OLMo-1B scale) https://arxiv.org/pdf/2603.26663
    - prov: web
  -
    - text: Centering and z-loss leave final loss unchanged (3.05 at 221M) and only reduce LR sensitivity https://arxiv.org/html/2601.02031v1
    - prov: web

open:
  -
    - q: Does the baseline row norm actually correlate with log frequency at step 100?
    - build: True
    - decides: per-token scale s included vs bias-only head
  -
    - q: What is the unigram entropy of FineWeb-Edu under GPT-2 BPE (for example 7.3 vs 7.6 nats)?
    - build: False
    - decides: kill window at steps 30–45 vs a later window
  -
    - q: What is the matched-seed noise of a 15-step mean loss across seeds, and can a 0.005 bias-only margin be resolved with 3 seeds?
    - build: True
    - decides: kill on steps 30–45 vs a 300-step window
  -
    - q: Should the bias be learnable after the counted init, or frozen as a buffer?
    - build: True
    - decides: parameter (counts toward c3) vs persistent buffer
  -
    - q: RMS vs L2 normalisation with matched s init: is there any difference beyond a constant?
    - build: False
    - decides: `f.rms_norm` rows vs `f.normalize` rows
  -
    - q: Does the self-logit of the current token drift after decoupling?
    - build: True
    - decides: plain readout vs readout minus tok(idx)
  -
    - q: Does a one-shard count differ meaningfully from a full-train count for rank-10k+ tokens?
    - build: True
    - decides: count from `data[0]` only vs from all shards (cost: startup seconds)

offshoots:
  - s06086e8ac4
  - s11a5cd9463
  - s6f95e7ea09
  - s1e1da9841a
  - s3d58dfadc3
  - s9490dc191f

objections:
  -
    - text: The idea says "RMS-normalized rows" with a zero-init log-scale, so the scale starts at exp(0)=1. An RMS-normalized row has L2 norm √384≈19.6, and the final `self.norm` output also has norm ≈19.6. Initial logits therefore have std ≈19.6, compared with ≈0.4 in the baseline (rows at std 0.02). Step-1 l
    - severity: high
    - cls: impl_slip
    - if_true: - metric: learn_z
      - op: below
      - threshold: 0.0
  -
    - text: Normalizing the head multiplies AdamW's effective head learning rate by about 1/rms(row)≈50×. Adam's sign-like step is ~3e-3 per element on rows whose elements are ~0.02, so each step rotates a row's output direction by ~15%. Under the tie, the input lookup sees the same rotation. Weight decay shrin
    - severity: high
    - cls: hidden_cost
    - if_true: - metric: trend
      - op: above
      - threshold: 0.0
  -
    - text: A zero-init bias under AdamW at lr 3e-3 with this schedule can move at most about Σlr ≈ 3e-3 × ~1300 effective steps ≈ 4 nats over the whole run. The unigram prior spans roughly 10+ nats between rank-100 and rank-10k tokens. The bias therefore cannot hold the prior within budget. The "push rare toke
    - severity: high
    - cls: interaction_cancels
    - link: L2
    - if_true: - metric: learn_z
      - op: below
      - threshold: 1.0
  -
    - text: `build_optimizers` sends every 1-D parameter to AdamW with weight_decay=0.1. That contradicts L1's "kept out of weight decay". A counted log-unigram bias would be decayed toward 0 and lose about 30% of its spread by step 1525, so the rows would have to relearn that part of the prior late in training
    - severity: medium
    - cls: impl_slip
    - link: L1
    - if_true: - metric: long_z
      - op: below
      - threshold: 2.33
  -
    - text: Adding the bias or scale after the matmul (`logits * scale.exp() + bias`) creates extra (4096×50304) tensors. Autocast does not cover elementwise ops, so bf16 logits + fp32 bias is promoted to an 0.82 GB fp32 logits tensor that is kept for backward. This is the copy the inline comment warns about, a
    - severity: high
    - cls: breaks_constraint
    - if_true: - metric: step_ratio
      - op: above
      - threshold: 1.15
  -
    - text: If the frozen-prior or counted-bias variant computes counts in `GPT.__init__` (or at startup) using a glob like `*.bin`, it picks up `val.bin` whenever it is present. That trips C6. If counts are instead stored as a non-persistent buffer, or the prior is added only inside the `targets is not None` b
    - severity: high
    - cls: impl_slip
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: The headline gain comes from the prior, and that is early-only. Baseline is already at 7.06 at step 26, close to unigram entropy, and the probe's per-10-step gain halves after step 10. A model that gets the prior for free leads at steps 1–100, then the baseline catches up as its rows encode frequenc
    - severity: high
    - cls: early_only
    - link: L5
    - if_true: - metric: trend
      - op: above
      - threshold: 0.0
  -
    - text: The premise that "decay shrinks rare rows to zero" is weak at this budget. AdamW decoupled decay gives a total factor of about exp(−3e-4 × ~1300) ≈ 0.68 on every row equally, not rare-specific. The input side renormalizes anyway: block 0's `n1` RMSNorm sees the 4-stream read of the raw embedding. La
    - severity: medium
    - cls: below_noise
    - link: L4
    - if_true: - metric: learn_z
      - op: below
      - threshold: 1.0
  -
    - text: The output gradient's radial component goes away, but its tangential component still encodes frequency. With exp(s_v) fixed near 1 early, the only way to lower a rare token's logit is to rotate ê_v against typical h. Row direction then correlates with frequency instead of row norm, and the "clean in
    - severity: medium
    - cls: interaction_cancels
    - link: L3
  -
    - text: Even if the mechanism works, the predicted 0.005–0.03 nats at the end sits mostly at or below what C7 can detect. The test needs >0.005 above the pooled mean at p<0.01 with a one-sided Welch test over 3 seeds. With seed std around 0.005–0.01, roughly a 0.015+ effect is required. The bias-only varian
    - severity: medium
    - cls: below_noise
    - if_true: - metric: long_z
      - op: below
      - threshold: 2.33
  -
    - text: Rare-token loss (12% of tokens, 11.6 nats) is the stated target, but those tokens are mostly not predictable from context at 50M tokens. A good unigram prior lowers their loss only toward their unigram cost. The baseline reaches that cost too by mid-run, so the exam (the mean over all tokens) moves 
    - severity: low
    - cls: metric_blind
    - if_true: - metric: long_z
      - op: below
      - threshold: 1.0
  -
    - text: The 100.6k new scalars are counted as non-embedding by `param_counts`, giving 21.34M, about +0.5%, well inside C3. The extra per-microbatch rms_norm of the 19.3M-element table, forward and backward, ×8 accumulation steps, is small next to the 79-GFLOP head matmul. This is not where it fails
    - severity: low
    - cls: hidden_cost
    - if_true: - metric: param_ratio
      - op: above
      - threshold: 1.1

premortem_verdict:
  Implemented as written, with a zero-init bias and RMS-normalized rows at unit scale, the head starts about 50× too hot and keeps a ~50× amplified effective learning rate. The zero-init bias cannot travel the ~10 nats of prior within the AdamW budget, so frequency is pushed back into row directions. 

## Asymmetric key/value expand widths  (picked_lane; part Sequence mixing: latent KV compression and expansion; inward; prior: new)
**Gated nonlinear latent** (leap, card research, p_success 0.2)
- mechanism: Add a 384 to 128 gate projection and set the latent to norm(W_c x times silu(W_g x)), so the KV code can select features conditionally on the token rather than being a fixed linear image of it. The whole K/V path is currently linear up to norms, which wastes the latent on features that only matter in combination.
- prediction: Parameters rise by 49k per layer, about 0.59M total, within the cap; step time up by under 3 percent; loss lower by 0.01 nats or more, with the largest gap on rare tokens whose K/V need context-dependent coding.
- test: Build-only parameter count, 3-step peak VRAM check since the gate adds a 128-wide activation per token, then a 45-step matched smoke run.
- breaks: Keys and values are linear functions of the normed residual read.
- keeps: 128-dim latent shared by K and V and by all heads, latent RMSNorm before expansion, separate q path, rotary path from x.
- kill: Mean loss over steps 40 to 45 not below the baseline's by at least 0.01 nats, or peak VRAM above 5.55 GB.
- target: loss on rare tokens, down
- scope: validation loss at 1024
- source: own

Other sketches in this group:
- Gate offset init to the linear code (): Initialise the gate pre-activation with a learned per-channel offset of +2, so silu(gate) is roughly a constant 1.76 at step 0 and the latent starts as the old linear code. Selectivity then grows only where gradients ask for it.
- Gate only layers 0–3 (): Apply the gate only in the four lowest layers, where token-identity-conditional features matter most and the layer-0 gradient spike sits. That uses 197k params instead of 590k.
- Layer-0-only gated latent (port): Apply the gate only in block 0, where the input is the bare token embedding. There the gate acts as a learned nonlinear per-token K/V code, decoupled from the tied embedding geometry. The other 11 layers stay linear.

## Successor-value heads  (picked_lane; part Sequence mixing: latent KV compression and expansion; inward; prior: new)
**Rope width sweep** (tune, card diagnosis, p_success None)
- mechanism: Position holds 32 of the 96 score dimensions with a single unnormalized shared rope key, a split copied from V4 and never tested at this width. Changing rd rebalances how much of the logit budget is positional versus content.
- prediction: at rd 16, parameters down 43k per layer and late-position loss up; at rd 48 or 64, parameters up 65k or 86k per layer and late-position loss down by 0.02 to 0.05 at step 100; validation effect in the range of 0.01 either way.
- test: sweep rd in {16, 48, 64}, 100 matched steps each with the position-bucket probe.
- target: loss at late positions.
- scope: validation loss at 1024
- source: own

## Rotary values with un-rotated readout  (picked_main; part Sequence mixing: decoupled rotary dims and QK-norm; inward; prior: new)
**Causal key depletion (overgrazing)** (leap, card research, p_success 0.1)
- mechanism: Shared structure: in a consumer-resource system consumers arriving in order deplete a shared resource, so a patch grazed by many earlier consumers yields less to later ones; here queries are consumers arriving in position order, keys are patches, and P[i,j] is consumption. Instantiation, in the last 2 layers only: S = q·k/sqrt(96), P0 = causal softmax(S), D = exclusive cumsum of P0 over queries, S' = S − λ_h·log1p(D), output = softmax(S')·v, with λ_h per head initialised at 0 so training starts exactly at baseline. Where it breaks: ecological resources regenerate and depletion is physical, whi
- prediction: peak VRAM rises by roughly 0.1 to 0.3 GB under activation checkpointing of those two attention cores; step time up 10 to 25 percent; deep-layer entropy up; loss lower by 0.005 to 0.015.
- test: first a 3-step smoke for peak VRAM and step time (must stay under 5.6 GB and 9.5 s), then 45 matched steps.
- breaks: a key's attractiveness is independent of how much attention earlier queries already took from it.
- keeps: softmax over keys, causal mask, QK-norm, RoPE, all other 10 layers unchanged.
- kill: VRAM over 5.6 GB at the 3-step check, or loss at 45 steps above baseline by more than 0.02.
- target: attention entropy in deep layers, up
- scope: validation loss at 1024
- source: own

### Interrogation

chain:
  -
    - id: L1
    - text: Rotate v[..., rotated slice] by R(ω·j) before SDPA, and rotate the same slice of the SDPA output by R(−ω·i) before `wo` -> each head's rotated slice of the output equals Σ_j a_ij R(ω(j−i)) v_j
    - quantity: unit check with forced one-hot attention at offset δ: the output should equal R(ωδ)v_j, and at δ=0 it should equal v exactly
    - baseline: n/a (identity today)
    - expect: max abs error under 2e-2 relative in bf16, and exactly 0 for the unrotated slice
    - breaks_if: the error exceeds 5e-2 at T=1024, or the δ=0 output is not v (sign or pairing bug)
  -
    - id: L2
    - text: For diffuse heads, the rotated slice cancels: Σ a_ij e^{iω(j−i)} shrinks for fast ω, so content in those dims is lost while offset information appears
    - quantity: per-layer ratio RMS(rotated slice of y) / RMS(unrotated slice of y)
    - baseline: unmeasured (≈1 by symmetry today)
    - expect: 0.2–0.7 in layers 0–2 (entropy 0.85–0.96), 0.6–1.0 in layers 6–10 (entropy 0.67–0.71)
    - breaks_if: the ratio is ≥0.95 in layer 0, which would mean no cancellation (and no offset signal) or a bug; or ≤0.3 in layers 6–11, which would mean the payload is destroy
  -
    - id: L3
    - text: The offset phase reaches the residual stream in a usable form -> a downstream probe can read the attended offset
    - quantity: linear probe from the block output to the argmax-attention offset (log-binned)
    - baseline: unmeasured
    - expect: probe accuracy up over baseline in layers ≥4, by any clear margin
    - breaks_if: the probe is no better than baseline (the phase is entangled with content and `wo`/MLP can't decode it)
  -
    - id: L4
    - text: Offset-aware reads help copying -> recall-token loss drops
    - quantity: recall-token loss (bigram already seen in window, 13% of targets)
    - baseline: 4.475 at step 100
    - expect: down 0.02–0.10 at step 100 (the idea's 4.30 target, −0.175, is optimistic)
    - breaks_if: the change is within seed noise, or recall loss rises
  -
    - id: L5
    - text: The cost of L2 stays small elsewhere -> non-recall loss is unchanged
    - quantity: non-recall loss
    - baseline: 6.704 at step 100
    - expect: within ±0.02
    - breaks_if: it rises more than 0.02 (lost value capacity outweighs offset info)
  -
    - id: L6
    - text: Net effect at the full budget
    - quantity: exam val loss at 50M tokens
    - baseline: pooled baseline mean (not in brief)
    - expect: −0.015 to +0.01
    - breaks_if: not below baseline by >0.005 at p<0.01 over 3 seeds

weakest:
  L4

qa:
  -
    - q: Does next-token prediction on recall tokens actually need the copy offset, or is it already implicit in which head did the copying?
    - a: In the standard induction circuit, the previous-token head always attends at offset 1, so its offset is implied by the head's identity. The induction head's prediction (the token after the earlier match) does not depend on how far back the match was. So the offset is mostly redundant for plain recall, and L4's gain probably comes from variable-offset heads, if it comes at all. This is a belief; it is unmeasured here.
    - decides: rotate every layer vs only layers whose heads attend at variable offsets (≥ mid-depth)
  -
    - q: Which frequencies should rotate the value: all 16 of the existing cache (32 dims) or a band?
    - a: `rope_cache(T, 32, 10000)` gives wavelengths 2π·10000^(k/16) for k = 0..15, from 6.3 up to about 35,300 tokens. - For k ≥ 13, the angle over the whole 1023-token window is under 0.6 rad, so those dims carry almost no offset information. - For k ≤ 2 (wavelengths 6–20), the phase aliases, and these dims cancel hardest under diffuse attention. - k = 3..10 covers wavelengths of about 35–1990 tokens, which is the "32 to 2048" range the first variant actually states. Note that variant's label is wrong: its "longest 8 wavelengths" would be k = 8..15, not 32–2048. I recommend the 16-dim band k = 3..10
    - decides: 32 dims with `cos, sin` as-is vs 16 dims using `cos[:, 3:11]`, `sin[:, 3:11]`
  -
    - q: How bad is the content loss from cancellation in the shallow, near-uniform layers?
    - a: With near-uniform attention over n tokens, |Σ a_ij e^{iωδ}| ≈ |sin(ωn/2)| / (n·sin(ω/2)), which is far below 1 except for the slowest ω. Layer 0 has entropy 0.957, so its rotated slice will carry close to zero content. Its attention gradient norm is already anomalous (0.24 against about 0.02 in its neighbours, z = 7 over 3 seeds), so perturbing layer 0's value path is the riskiest place. That last point is a belief.
    - decides: apply in all 12 layers vs skip layers 0–2
  -
    - q: Is the inverse rotation exactly what `apply_rope` can do, and is the pairing right?
    - a: `apply_rope` pairs dim i with dim i+D/2 inside the tensor it is given, and slices `cos[:T]`. So these calls give the exact inverse at query position i: - `apply_rope(v[..., :32], cos, sin)` for the forward rotation - `apply_rope(y[..., :32], cos, -sin)` for the inverse For the 16-dim band, call it on a 16-dim slice with `cos[:, 3:11]` and `sin[:, 3:11]` (8 pairs). Apply it to `y` while `y` is still (B, H, T, 64), before the transpose. This is read from the script.
    - decides: reuse `apply_rope(y, cos, -sin)` vs write a new helper
  -
    - q: Can `wo` (linear) and the MLP actually decode offset from R(ωδ)v_j?
    - a: The output is bilinear in content and offset. A linear readout recovers the offset only along directions where v_j is consistent across tokens. Values have no bias, and latent RMSNorm does not guarantee such a direction, so decoding depends on the downstream SwiGLU. This is unknown empirically; the build stays faithful (content rotation only), and the carrier version is listed below as a separate idea.
    - decides: rotate content only (the idea as written) vs also add a constant carrier pair whose rotated output is a pure offset spectrum
  -
    - q: Is a 45-step kill check long enough to see a recall effect, and is 0.03 nats above seed noise?
    - a: Unknown. 45 steps is about 1.47M tokens. Much of the 2.2-nat recall advantage at step 100 is likely repeated-bigram frequency rather than induction, and whether induction heads have formed by 1.5M tokens at 21M params is unmeasured. The seed-to-seed spread of recall loss is not in the brief. Measure it on 3 baseline seeds before using 0.03 as a kill threshold.
    - decides: kill at 45 steps vs extend the check to 100 steps (or a 10M-token mid-run)
  -
    - q: What do step time and memory cost?
    - a: The extra work is two elementwise rotations per layer on tensors of (4, 6, 1024, ≤32), which is negligible next to the attention and MLP matmuls. Saved activations add roughly 3 MB per layer at bf16, about 40 MB total, well under the 5.6 GB cap. +1% step time is plausible. Use `torch.cat` rather than in-place writes so autograd stays simple. This is a belief from the arithmetic.
    - decides: inline slicing and `torch.cat` vs in-place slice assignment to avoid copies
  -
    - q: Is there prior evidence for rotating values and un-rotating outputs?
    - a: Not from the brief, and none of its sources tests it. My recollection is that Su Jianlin's blog discussed "VO-RoPE" (rotating V and O) in 2025 without a clear LM gain, but I could not verify a URL, so treat this as unverified.
    - decides: build as a leap vs deprioritize
  -
    - q: Does the change survive the harness exam, which builds `GPT(cfg)` from the saved config?
    - a: No parameters or buffers are added, so `model.pt`'s `state_dict` is unchanged. A new `Config` field is only safe if the exam rebuilds `Config` from this file. Hardcode the constants in `Attention`, as `rd = 32` already is.
    - decides: new `Config` fields (`v_rope_dims`, band) vs hardcoded constants in `Attention`
  -
    - q: Should value phases be per-head, given the rotary key is shared across heads?
    - a: The script uses one cache for all heads, so all heads share the value frequencies, and heads differ only through their attention patterns. Per-head frequencies are a separate idea (STRENGTHEN below), not part of the faithful build. This is from the script.
    - decides: one shared cos/sin for all heads vs per-head frequency offsets

known_tagged:
  -
    - text: Head dims: content q/k 64, rotary 32 (shared key `kr` broadcast to 6 heads), v 64; SDPA scale is the default 1/sqrt(96) because q/k are 96-wide
    - prov: script
  -
    - text: `v` is the second half of each head's 128-wide slice of `expand(latent_norm(compress(x)))`; the rotated slice is `v[..., :32]` (or a 16-dim band) taken before SDPA
    - prov: script
  -
    - text: `cos`/`sin` come from `rope_cache(T, 32, theta)` with shape (T, 16) and are built once in `GPT.forward`; reuse them, with no new buffers
    - prov: script
  -
    - text: The inverse rotation is `apply_rope(y_slice, cos, -sin)` applied to `y` in (B, H, T, 64) layout before `transpose(1, 2)` and `wo`
    - prov: script
  -
    - text: Wavelengths of the 16 frequencies run from 2π to 2π·10000^(15/16) ≈ 35,300 tokens; k = 3..10 spans about 35–1990 tokens
    - prov: script
  -
    - text: Attention entropy by layer runs from 0.957 (layer 0) to 0.667 (layer 9); diffuse layers will cancel the rotated content
    - prov: measured
  -
    - text: Recall tokens are 13% of targets, with loss 4.475 against 6.704 for the rest at step 100
    - prov: measured
  -
    - text: Layer-0 attention gradient is anomalously large (0.29 against a neighbour-predicted ~0.04, 3 seeds); watch it after the change
    - prov: measured
  -
    - text: No new parameters and no init changes; Muon/AdamW grouping is unaffected
    - prov: script
  -
    - text: The sliding-window branch is unaffected; the rotation sits outside SDPA, so both branches need it
    - prov: script
  -
    - text: Seed noise on recall-token loss is unmeasured; get it before trusting a 0.03 threshold
    - prov: belief

open:
  -
    - q: What is the seed-to-seed std of recall-token loss at steps 45 and 100?
    - build: False
    - decides: kill threshold 0.03 vs a noise-derived threshold
  -
    - q: Have induction heads formed by 1.5M or 3.3M tokens at this scale?
    - build: True
    - decides: kill check at 45 steps vs 100 steps or mid-run
  -
    - q: Does `wo` plus SwiGLU decode phase from rotated content, or does it need a carrier?
    - build: True
    - decides: content rotation only vs adding a carrier pair
  -
    - q: Is the full 32-dim/16-frequency version better or worse than the k = 3..10 band?
    - build: True
    - decides: 32 rotated dims vs 16
  -
    - q: Does the exam rebuild `Config` from this file or from its own copy?
    - build: True
    - decides: new `config` fields vs hardcoded constants
  -
    - q: Is there a published LM ablation of rotating V and O (VO-RoPE), and what did it find?
    - build: False
    - decides: run as a leap vs deprioritize
  -
    - q: Does damping layer 0's value content reduce or worsen its gradient anomaly?
    - build: True
    - decides: include layer 0 vs exclude it

offshoots:
  - sdce62f76f0
  - sce278c5f21
  - sf2547da1ce
  - sbe9e85c46f
  - sd480bde265

objections:
  -
    - text: Fast rotary pairs scramble the copied content.** The script's `rope_cache(T, 32, 10000)` gives the rotated slice wavelengths of 2π·10000^(k/16): 6.3, 11, 20, 35 and 63 tokens for k=0–4. An induction head copies a token from an offset that varies from window to window. In those pairs the copied value
    - severity: high
    - cls: interaction_cancels
    - link: L4
    - if_true: - metric: learn_z
      - op: above
      - threshold: 0.0
  -
    - text: The phase cannot be decoded without the content.** The output is R(ωδ)v_j. Reading δ from it requires knowing v_j, and v_j varies across tokens. `wo` is linear and the SwiGLU MLP is shallow, so neither can separate content from phase except through products of pairs. The offset signal therefore stay
    - severity: high
    - cls: metric_blind
    - link: L3
    - if_true: - metric: long_z
      - op: above
      - threshold: -1.0
  -
    - text: The cost lands on layer 0, the most important attention layer.** Layers 0–2 have entropy 0.85–0.96. With attention that diffuse, Σa·e^{iωδ} averages the fast pairs toward zero, so half the value dims in those layers become noise. Layer 0's attention gradient is the run's biggest anomaly (0.29, z 7.0
    - severity: high
    - cls: interaction_cancels
    - link: L5
    - if_true: - metric: learn_z
      - op: above
      - threshold: 0.0
  -
    - text: The effect is too small to pass the test.** Recall tokens are 13% of exam targets. Even the chain's own expected gain of −0.02 to −0.10 on recall is only −0.003 to −0.013 nats overall, before any non-recall cost. C7 needs a margin of more than 0.005 at p<0.01 with 3 seeds. With seed standard deviati
    - severity: high
    - cls: below_noise
    - link: L6
    - if_true: - metric: long_z
      - op: above
      - threshold: -2.5
  -
    - text: The early advantage disappears.** At step 100 the baseline has no mature induction circuit, and recall loss is 4.48. Explicit offset phase might give a head start. Over the full run (50M/32768 ≈ 1525 steps), standard previous-token-plus-induction circuits form through the shared rotary key and the 4
    - severity: medium
    - cls: early_only
    - link: L4
    - if_true: - metric: trend
      - op: above
      - threshold: 0.0
  -
    - text: Implementation slip: the inverse rotation is applied after transposing.** `apply_rope` slices `cos[: x.size(2)]`. A builder who inverse-rotates `y` after `y.transpose(1, 2)`, which has shape (B,T,H,D), gets `cos[:6]` of shape (6,16). That broadcasts silently against (B,T,6,16), so the output is rota
    - severity: high
    - cls: impl_slip
    - link: L1
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: Implementation slip: the inverse rotation does not negate sine.** Reusing `apply_rope(y, cos, sin)` for the readout, instead of `apply_rope(y, cos, -sin)`, gives R(ω(i+j)) rather than R(ω(j−i)). That puts absolute position into half of each value head, and the model degrades at positions it rarely s
    - severity: medium
    - cls: impl_slip
    - link: L1
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: The wrong variant was chosen.** The idea text says "first 32 of 64". Only a narrower slice, about 16 dims with wavelengths of 600 to 35k (k≥8), would carry offset at a scale useful within a 1024 context without scrambling content. Building the "first 32" version faithfully includes all 8 fast pairs:
    - severity: medium
    - cls: impl_slip
    - link: L2
    - if_true: - metric: learn_z
      - op: above
      - threshold: 0.0
  -
    - text: The step-time estimate is optimistic.** SDPA cannot fuse the rotation, so each of the 12 layers adds two rotations on (4,6,1024,32) tensors. Each `cat` allocates fresh tensors, and backward saves them. On a 4050 already bound by the 4-stream einsums, the realistic overhead is +3–6%, not +1%. That st
    - severity: low
    - cls: hidden_cost
    - if_true: - metric: step_ratio
      - op: above
      - threshold: 1.05
  -
    - text: bf16 rotation adds noise to half of every value head.** `apply_rope` casts cos/sin to bf16. At positions near 1000, rotating forward then back accumulates relative error of around 2^-8 on every value, including δ=0 self-copies. That adds small noise across all 12 layers, which works against the late
    - severity: low
    - cls: hidden_cost
    - link: L1
  -
    - text: The test metric misses the actual effect.** The KILL criterion uses recall-token loss at 45 steps. If the matched smoke is not paired by seed, recall-loss noise at that point is probably larger than 0.03, so the gate can pass on chance. The exam averages over all tokens, where the non-recall cost do
    - severity: medium
    - cls: metric_blind
    - link: L4
    - if_true: - metric: long_z
      - op: above
      - threshold: 0.0

premortem_verdict:
  Rotating the fast-wavelength value pairs makes the copied content arrive at an offset-dependent phase that `wo` cannot undo, so induction loses value bandwidth and diffuse early layers (layer 0 especially) lose payload, while the offset signal stays entangled with content and undecodable; the net ex

## Shifted-key induction heads  (picked_main; part Sequence mixing: head count, head structure and output proje; outward; prior: new)
**Shifted-key heads (one-hop induction)** (leap, card research, p_success 0.18)
- mechanism: For heads 3 to 5, the key at position j is the (post-RoPE) key of token j−1 while the value stays token j's; a query that matches "the token before me last time" directly reads "what followed it". Today this needs two layers (a previous-token head feeding an induction head); fusing it frees those layers for other circuits and should lower loss on repeated spans within 1024 tokens.
- prediction: Zero parameter change, step time within 1%; training loss separates from baseline after the induction bump (roughly step 150 to 300), gap of 0.02 to 0.04 nats at 50M tokens, largest on late positions.
- test: 45-step matched smoke run on seed 0; 3-step timing; unit check that query i never sees token i+1 (compare logits for a sequence truncated at i).
- breaks: The key a query scores and the value it reads belong to the same token.
- keeps: softmax causal attention, the latent KV path, QK-norm, RoPE on the rotary slice, the same wo and head count.
- kill: mean loss over steps 40 to 45 more than 0.02 nats above the baseline's at the same steps.
- target: recall-token loss, down; loss at late positions, down
- scope: validation loss at 1024
- source: own

Other sketches in this group:
- Bigram-prefix shifted heads (leap): In heads 3–5, content dims 0–31 of the key come from token j−1 and dims 32–63 from token j−2. Query dims 0–31 come from token i and dims 32–63 from token i−1. The head then matches a two-token prefix, which cuts false matches on frequent tokens, with zero parameters.
- One induction head in layers 1–3 (leap): Shift the key only for head 5 in layers 1, 2 and 3. All other heads stay standard, so 33 of 36 standard-head slots are kept and the L4 cost is small.
- Learned key smear (port): Drop the hard shift. For every head, k'_j = k_j + α_h·k_{j−1}, with a per-head scalar α initialised at 0 (6 params per layer), applied after `k_norm`. Each head chooses how much predecessor identity its keys carry.

### Interrogation

chain:
  -
    - id: L1
    - text: Heads 3–5 use the key of token j−1 with the value of token j, so in one layer they can attend to positions whose predecessor matches the current token
    - quantity: share of attention mass that heads 3–5 put on "successor of an earlier copy of token i" positions, per layer
    - baseline: unmeasured
    - expect: up, from chance (<0.05) to >0.2 in at least one layer by step ~300
    - breaks_if: this share stays near chance in every layer through step 500.
  -
    - id: L2
    - text: One-hop induction lowers loss on repeated spans early in training
    - quantity: recall-token loss (targets whose bigram was already seen in the window)
    - baseline: 4.475 at step 100 (vs 6.704 for other tokens)
    - expect: down 0.15–0.3 nats against baseline at matched steps 200–1525
    - breaks_if: the gap at matched steps is under 0.05 nats.
  -
    - id: L3
    - text: Recall gains build up with context length
    - quantity: loss at 256–1024 minus loss at 0–64
    - baseline: +0.459 at step 100
    - expect: this difference shrinks against baseline by 0.02–0.1
    - breaks_if: the late-minus-early difference matches baseline within 0.01.
  -
    - id: L4
    - text: Taking 3 of 6 standard heads away in all 12 layers costs nothing on the other 87% of targets
    - quantity: non-recall token loss
    - baseline: 6.704 at step 100
    - expect: within ±0.01 of baseline
    - breaks_if: it rises by more than ~0.15 × the recall-token gain, which cancels the net effect.
  -
    - id: L5
    - text: Net effect on the exam
    - quantity: exam val loss at 50M tokens
    - baseline: pooled baseline mean (not in the measurements above)
    - expect: down 0.01–0.04
    - breaks_if: the gap is ≤0.005 or one-sided Welch p ≥ 0.01 over 3 seeds.

weakest:
  L4

qa:
  -
    - q: Is the predicted 0.02–0.04 nats arithmetically reachable through recall tokens alone?
    - a: Recall tokens are 13% of targets (measured). A 0.02 net gain needs about 0.15 nats on them, and 0.04 needs about 0.31 nats, assuming zero cost elsewhere. That is a large recall gain, so the build cannot also afford a 0.01 loss on the other 87% of targets. This argues for logging both groups from the first run.
    - decides: 3 shifted heads per layer vs 1 shifted head per layer (lower cost to L4)
  -
    - q: How should the shifted key at position 0, which has no predecessor, be filled?
    - a: `torch.roll` wraps token T−1's key into position 0. That puts future content into a key every query can see, so it is a leak and must not be used. The safe choice is a zero key: `cat([zeros_like(k[:,:,:1]), k[:,:,:-1]], dim=2)`. It gives logit 0, so query 0 attends only to j=0 and reads its own value (no NaN). Watch whether position 0 becomes a sink for heads 3–5; baseline first-token mass is 0.002–0.006.
    - decides: a zero key vs `torch.roll` (or a duplicate of token 0's key)
  -
    - q: Should the whole key be shifted (content plus rotary, post-RoPE), or only the content slice with rotary at j?
    - a: The idea specifies post-RoPE, so shift the whole key. The rotary score then sees relative offset i−j+1, and purely positional heads still work, offset by one. `kr` is a single head expanded to all heads, so the simplest faithful build shifts the final `k` tensor for heads 3:6 after the `torch.cat`. Both variants are causal.
    - decides: shift the final concatenated `k[:, 3:6]` vs shift `self.k_norm(k)` only and keep `kr` at j
  -
    - q: Can QK-norm logits get sharp enough to pick one matching token out of ~1000?
    - a: With norm weights at 1, the content logit is at most 8·8/√96 ≈ 6.5 (SDPA scale is 1/√96 because q has 96 dims). Selecting one position among ~1000 needs a gap of about ln(1000) ≈ 6.9. The baseline already grows its max logit to 7.8–22.75 by step 100 (measured), so the norm gains learn this quickly. Keep init 1 for fidelity.
    - decides: keep the `q_norm`/`k_norm` weight init at 1 vs raise it (e.g. 2)
  -
    - q: Does induction even form inside 50M tokens (1525 steps), and when?
    - a: Unknown for this recipe. Olsson et al. report the induction phase change in small models at roughly 2.5–5B tokens with large batches (https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). This setup has a different optimizer, batch and LR, so that does not transfer. At step 45 (1.47M tokens) neither model likely has induction, so the 45-step kill tests only harm, not benefit.
    - decides: trust the 45-step kill vs add a matched 300-step run before the 3-seed run
  -
    - q: Which layers should get shifted heads?
    - a: Unknown. The mechanism needs the key to reflect token j−1's identity, which is cleanest in early layers where the residual is close to the embedding. Later layers already mix in neighbour information through standard heads. Build it as specified, but log the L1 attention share per layer so a specialised version can be chosen from data.
    - decides: heads 3–5 in all 12 layers (as specified) vs only in layers 0–5
  -
    - q: Does it fit C3 and C4?
    - a: It adds no parameters, so non-embedding stays at 21.24M. Each layer holds an extra (4,3,1024,96) bf16 tensor, about 2.4 MB, roughly 28 MB across 12 layers; that is negligible against 5.6 GB. SDPA keeps `is_causal=True`, so the fused kernel path is unchanged. A `cat` of a slice should stay within ~1% step time, but measure it in the 3-step timing check.
    - decides: `cat`-based shift inside `Attention.forward` vs a precomputed index gather
  -
    - q: Can the exam harness rebuild the shifted model from `model.pt`?
    - a: Unknown how `nano/exam.py` maps the saved config onto `Config`. Hardcoding (heads `n_head//2:` shifted) reproduces the model whatever the harness does with extra keys. A field with a default equal to the new behaviour is also safe if the exam instantiates this file's `Config`.
    - decides: hardcode the shift vs add a Config field (e.g. `shift_heads: int = 3`)
  -
    - q: Does the layer-0 gradient anomaly (10× attention grad) interact with the shift?
    - a: Unknown. Layer 0 is where one-hop induction is most natural, which is an argument for keeping it. Log layer-0 attention grad norm (baseline 0.2406 at step 100) and see whether it moves.
    - decides: shift in layer 0 vs start shifting at layer 1

known_tagged:
  -
    - text: Heads 3–5 means `k[:, 3:6]` in the `(B, H, T, 96)` key tensor after `k = torch.cat([self.k_norm(k), kr.expand(...)], dim=-1)`; values `v` are not touched.
    - prov: script
  -
    - text: `kr` comes from one rotary projection shared by all heads (`self.rotary` → `rd=32`, expanded to `n_head`), so shifting it for heads 3–5 only must act on the expanded tensor.
    - prov: script
  -
    - text: `k_norm` is per-token, so shifting before or after it gives the same result, except for the padded position.
    - prov: script
  -
    - text: `torch.roll` along T leaks token T−1 into position 0's key; pad with zeros instead.
    - prov: belief
  -
    - text: SDPA scale is 1/√96 (q last dim = 64 + 32); the content-logit ceiling at norm weight 1 is about 6.5.
    - prov: script
  -
    - text: Total steps at 50M tokens = 50e6 // 32768 = 1525; the predicted induction window (steps 150–300) is 4.9–9.8M tokens.
    - prov: script
  -
    - text: A 45-step smoke run has its own schedule: warmup is 1 step and cooldown starts at step 31. Baseline and candidate must use the same `--smoke-steps`.
    - prov: script
  -
    - text: Recall tokens are 13% of targets with loss 4.475 vs 6.704 at step 100; late-minus-early position loss is +0.459.
    - prov: measured
  -
    - text: Max attention logit at step 100 is 7.81 (layer 0) to 22.75 (layer 8); first-token attention mass is 0.0023–0.0059.
    - prov: measured
  -
    - text: The leak test should check that logits at positions ≤ i are bit-identical (in fp32) when tokens > i are randomised; include i = 0 and i = T−1.
    - prov: belief
  -
    - text: H3 put a shift SSM on keys specifically so one layer can do associative recall/induction (https://arxiv.org/abs/2212.14052).
    - prov: web
  -
    - text: Primer's depthwise convolution after the Q/K/V projections (a learned local shift) improved LM quality (https://arxiv.org/abs/2109.08668).
    - prov: web
  -
    - text: One-layer attention-only transformers cannot form induction heads without a previous-token mechanism; normally a previous-token head and an induction head span two layers (https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
    - prov: web
  -
    - text: No parameter change: non-embedding stays at 21.24M.
    - prov: script

open:
  -
    - q: Does the baseline form induction heads at all within 1525 steps, and at which step?
    - build: True
    - decides: trust the 45-step kill alone vs add a matched 300-step gate before seeds
  -
    - q: Is the cost on non-recall tokens from losing three standard heads per layer below 0.01 nats?
    - build: True
    - decides: 3 shifted heads per layer vs 1 per layer
  -
    - q: Which layers' shifted heads actually do induction (the L1 share per layer)?
    - build: True
    - decides: all 12 layers vs early layers only
  -
    - q: Does the zero-padded key at position 0 become an attention sink for heads 3–5?
    - build: True
    - decides: zero pad vs a learned null key
  -
    - q: How does `nano/exam.py` rebuild `Config` from `model.pt`?
    - build: True
    - decides: hardcoded shift vs a new config field
  -
    - q: What is the seed-to-seed std of the exam val loss? Is a 0.02 gain detectable at p < 0.01 with 3 seeds?
    - build: False
    - decides: 3 seeds vs 5 seeds
  -
    - q: Does the layer-0 attention gradient anomaly change when layer 0's heads 3–5 are shifted?
    - build: False
    - decides: include layer 0 vs start at layer 1

offshoots:
  - sc21d6f35ef
  - s8302812320
  - s40dab6db58
  - s545c7c8c3a
  - s0ab97a5c0e
  - sf23c585ac6

## Slow EMA AGC on MLP write  (picked_main; part Per-token computation: SwiGLU MLP; outward; prior: new)
**Slow automatic gain control on the MLP write** (leap, card research, p_success 0.15)
- mechanism: Shared structure: a radio receiver chain has amplifier stages whose output envelope is measured over a window far longer than the signal, and a feedback gain keeps each stage inside the next stage's linear range; our chain is 12 blocks writing into a shared stream, the "signal" is the token-to-token variation of a write, the "envelope" is the batch-level RMS, and the stuck measurement is an MLP write RMS that drifts to 10.9 because magnitude can only be set by inflating three Muon-trained matrices against weight decay. Instantiation: the MLP keeps a detached running RMS of its own output (mome
- prediction: MLP output RMS stays within 2x of its initial per-layer level instead of growing to 10.9; matrix weight norms grow less; step time within 1%; parameters +12 and +12 buffers.
- test: 45 matched steps against the baseline curve; log per-layer write RMS and exp(a) every 10 steps; a 3-step smoke to confirm peak VRAM is unchanged.
- breaks: the magnitude of a sublayer's write into the residual is decided by the norms of its weight matrices.
- keeps: SwiGLU with its three matrices, widths, Muon on the matrices, the scaled init, the 4-stream combine.
- kill: mean loss over steps 40 to 45 more than 0.03 nats above the baseline's at the same steps, or any layer's gain exp(a)/ema outside [0.02, 50].
- target: residual RMS growth, down; update-norm spread, down.
- scope: validation loss at 1024
- source: own

Other sketches in this group:
- Two-timescale AGC with a fast log-gain (): Put `a` in its own AdamW group (lr 3e-2, wd 0) and use a micro-batch EMA with τ ≈ 1 step. Initialise exp(a) so that c = 1 at step 0, letting `a` cover about 2.5 nats by step 100.

### Interrogation

chain:
  -
    - id: L1
    - text: Scaling the MLP output by c = exp(a)/ema_rms (ema detached) makes each layer's MLP write RMS follow exp(a) rather than the matrix norms
    - quantity: per-layer MLP write RMS before `combine`, divided by exp(a)
    - baseline: unmeasured (the 1.403→10.893 column is the residual stream after each block, not the MLP write)
    - expect: ratio within [0.5, 2] once the EMA settles; overshoot during warmup of e^{r·τ} (r = raw growth rate per step, τ = EMA lag in steps)
    - breaks_if: ratio stays outside [0.5, 2] after step 20, or tracks raw
  -
    - id: L2
    - text: Bounded MLP writes → less residual-stream growth across depth
    - quantity: residual RMS after block 11 / after block 0
    - baseline: x7.8 (1.403→10.893 at step 100)
    - expect: down to roughly x3–x6, but only if MLP writes are most of the stream
    - breaks_if: growth stays ≥ x7 (attention writes dominate, and they are untouched).
  -
    - id: L3
    - text: Write scale no longer depends on
    - baseline: median 0.02009, max 0.03265; weight norms unmeasured
    - expect: MLP
    - breaks_if: MLP
  -
    - id: L4
    - text: Smaller stream growth and steadier updates → equal or lower loss
    - quantity: mean loss over steps 40–45 (same total-step schedule as the baseline); final exam val loss
    - baseline: about 6.6–6.7 at steps 40–45 (6.59 at step 51); final is the pooled baseline mean
    - expect: within +0.03 at steps 40–45; final in the band −0.005 to −0.03
    - breaks_if: more than +0.03 at steps 40–45, or the final value does not beat the pooled mean by more than 0.005 at p<0.01.

weakest:
  L4

qa:
  -
    - q: In this pre-norm stack, can absolute residual RMS affect loss at all, or only the ratio of each write to the accumulated stream?
    - a: Every read goes through an RMSNorm (`n1`, `n2`, final `norm`), so a global rescale of the stream does not change the logits. Only the relative share of each write matters: the share of late layers shrinks as the stream grows. Whether that shrinking costs loss at this scale is unknown. If the mechanism is real, the relative target is the one that acts on it.
    - decides: absolute target (write RMS = exp(a)) vs relative target (write RMS = exp(a)·ema of block-input stream RMS)
  -
    - q: Under AGC, will Muon grow the MLP matrices less (as predicted) or more?
    - a: Probably more. With a detached EMA, each step's gradient still contains a "make the output bigger" radial component. Muon applies fixed-size updates regardless of gradient scale, so ||W|| keeps growing while the EMA cancels the effect on the output. That is the scale-invariant norm-growth effect, with effective step size falling as ||W|| grows (https://arxiv.org/abs/1803.01814, https://arxiv.org/abs/1706.05350). Weight decay 0.1·lr is the only brake. This is unmeasured, so log ||W||.
    - decides: leave W free vs add a norm constraint or extra weight decay on the MLP matrices
  -
    - q: Can a move far enough under the shared AdamW group (lr 3e-3, wd 0.1)?
    - a: Probably not. Adam moves a by at most about lr per step: roughly 0.12 nats over 45 steps and about 4 nats over the whole 1525-step run. In the baseline, writes go from about 0.01 (init) to O(1) by step 100, about 4.6 nats. A gain-1 init would leave the MLP nearly silenced early while attention writes grow freely. Give a its own group with lr around 0.03–0.1 and no weight decay. Weight decay would also pull a toward 0, i.e. gain 1/ema.
    - decides: a in the existing "other" group vs its own group (higher lr, no weight decay)
  -
    - q: How should a and the EMA be initialised?
    - a: Estimated initial write RMS is about 0.01. Inputs are normalized to RMS 1; w_gate/w_up std 0.02 gives pre-activation std of about 0.39; silu·up gives about 0.077; w_down std 0.00408 over 1077 inputs gives about 0.010. So a = 0 means gain ≈ 100, which trips the idea's own kill bound (>50) at step 1. Use lazy init: a sentinel EMA value replaced via `torch.where` on the first training forward (no `.item()` sync), and a = log(target), where the target comes from Q3/Q11.
    - decides: a = 0 vs a = log(first measured write RMS); EMA analytic init vs lazy init on the first training forward
  -
    - q: What EMA momentum, and updated per micro-batch or per optimizer step?
    - a: Early Muon relative updates are large. For w_down: lr 0.02 × ||orthogonal update||_F ≈ √384 ≈ 19.6 gives 0.39 per step against an init norm of 2.6, about 15% per step. Growth at rate r with lag τ leaves a constant overshoot of e^{rτ}. A 12-step window (0.99 per micro-batch with 8 micro-batches per step) could overshoot several-fold during warmup. Use about 0.9 per micro-batch, or bias-corrected momentum that ramps up. "Slow" is safe only after the warmup phase.
    - decides: momentum 0.9 vs 0.99 per micro-batch; update every micro-batch vs once per step
  -
    - q: Does the exam call `model.eval()` or run under `no_grad`?
    - a: Unknown. Gate on both, and register the EMA as a persistent buffer so `model.pt` carries it. Otherwise the exam would update statistics on validation tokens and make results depend on batch order.
    - decides: gate EMA updates on `self.training` only vs on `self.training and torch.is_grad_enabled()`
  -
    - q: Should the current batch's statistic enter the scale used for that same batch?
    - a: Update-then-use computes the RMS over all 1024 positions, so future tokens leak a tiny amount into earlier positions' scale. That is a causal violation, even if small, and train and eval would differ. Use the previous EMA, then update.
    - decides: use the previous EMA, then update vs update, then use
  -
    - q: Does the idea's kill bound on gain, exp(a)/ema ∈ [0.02, 50], fire by construction?
    - a: Likely. Raw output scales roughly with the product of three matrix norms. By step 45, raw growth is somewhere between about 1.6x (random-walk updates) and about 45x (aligned updates). The baseline reaching residual RMS 1.4 by step 100 suggests the aligned end. With a pinned (Q3), the gain falls toward or below 0.02 with the output still healthy. The bound measures W growth, not failure.
    - decides: keep the absolute-gain kill vs redefine it as gain relative to its step-0 value, or as write RMS / exp(a)
  -
    - q: Are MLP writes or attention writes the bigger part of the stream growth?
    - a: Unmeasured. Layer 0 shows anomalous gradients in both sublayers (attention 0.29, MLP 0.23, z 7.0 and 5.8). Log per-sublayer write RMS in the baseline before building; if attention dominates, L2 cannot hold.
    - decides: AGC on the MLP only (as specified) vs MLP plus attention
  -
    - q: How should the 45-step test match the baseline?
    - a: `lr_mult` scales with total steps. At 45 steps, warmup is 1 step and cooldown starts at step 31. At 100 steps, warmup is 2 steps and cooldown starts at 70. A 45-step run is therefore not matched to the 100-step baseline curve. Run 100 steps (or rerun the baseline at 45) and compare steps 40–45.
    - decides: `--smoke-steps 45` vs a 100-step run read at steps 40–45
  -
    - q: Should the EMA track mean-square or RMS, and in what precision?
    - a: Track sqrt(ema(mean y²)) in fp32. It is the unbiased envelope and avoids bf16 rounding across 12 layers × 8 micro-batches. Compute exp(a)/ema in fp32 and cast the scalar to `y.dtype`.
    - decides: sqrt(ema(mean y²)) in fp32 vs ema(rms) in bf16

known_tagged:
  -
    - text: Plug point: `MLP.forward`, after `w_down`, before `Block.combine` applies `2*sigmoid(write_logits)` (range 0–2) per stream.
    - prov: script
  -
    - text: The MLP write scale is already reachable without the matrices: the `n2.weight` gain, `read_logits`, and `write_logits` (capped at 2x), so the BREAKS premise is only partly true.
    - prov: script
  -
    - text: The 1.403→10.893 column is "Residual stream RMS after each block", not the MLP output RMS that the idea's mechanism cites.
    - prov: measured
  -
    - text: Residual RMS increments per layer are nearly constant (about 1.3 early, about 0.65 late), i.e. roughly linear growth, which suggests correlated writes rather than random ones.
    - prov: measured
  -
    - text: A 0-dim `a` falls into the AdamW "other" group (lr 3e-3, wd 0.1) unless split out.
    - prov: script
  -
    - text: Full run is 1525 optimizer steps, warmup 30, cooldown from step 1067, 8 micro-batches of 4×1024 per step.
    - prov: script
  -
    - text: Muon normalizes gradient scale, so c rescales W gradients without changing Muon's step size; it does change the global `clip_grad_norm_` factor seen by the Adam parameters.
    - prov: script
  -
    - text: Initial MLP write RMS is about 0.01 (derivation in A4).
    - prov: belief
  -
    - text: Muon per-step ||ΔW||_F is about 0.39 for w_down and about 0.66 for w_gate/w_up at full lr, against init norms of 2.6 and 12.9.
    - prov: belief
  -
    - text: Buffers are excluded from `param_counts`; +12 parameters is negligible against C3.
    - prov: script
  -
    - text: Scale-invariant weights grow in norm under gradient training and their effective lr decays: https://arxiv.org/abs/1803.01814
    - prov: web
  -
    - text: Peri-LN (normalizing sublayer outputs) bounds activation variance; its per-token output normalization is the known technique this idea must stay distinct from: https://arxiv.org/pdf/2502.02732
    - prov: web
  -
    - text: Learned scalar branch gains (ReZero, LayerScale) are the known non-normalizing relatives: https://arxiv.org/abs/2003.04887, https://arxiv.org/abs/2103.17239
    - prov: web

open:
  -
    - q: What is the per-layer MLP write RMS vs attention write RMS in the baseline at steps 0, 45, 100 and 1525?
    - build: True
    - decides: mlp-only agc vs mlp plus attention
  -
    - q: Does `nano/exam.py` call `model.eval()` and/or `torch.no_grad()`?
    - build: True
    - decides: ema update gated on `self.training` vs on `self.training and grad enabled`
  -
    - q: Does residual growth across depth measurably cost loss at this scale, e.g. via a frozen-ratio control run?
    - build: True
    - decides: absolute target vs relative-to-stream target
  -
    - q: Do MLP weight norms grow faster under AGC with Muon (the ratchet)?
    - build: True
    - decides: w left free vs norm projection or extra weight decay
  -
    - q: Which lr for a lets it track write growth without oscillating against the EMA lag?
    - build: True
    - decides: a in the shared adam group vs its own group with lr 0.03–0.1
  -
    - q: Is the EMA momentum that is safe during warmup (about 0.9 per micro-batch) too noisy later at batch-level RMS?
    - build: True
    - decides: fixed momentum vs a momentum schedule
  -
    - q: Is the near-linear residual growth caused by a shared DC direction in the writes?
    - build: False
    - decides: n/a

offshoots:
  - s92664facb8
  - s58d8a01d0d
  - s5b0cbe54a5
  - s912b21b5a6
  - s18916319a9
  - s04e31b5178

objections:
  -
    - text: Muon removes the cost L3 claims to free.** Each Muon step applies `lr*scale*NS5(g)`, a fixed-size orthogonalized update, no matter whether the loss "wants" larger weights. Matrix norms in both arms are set by the Muon/weight-decay equilibrium (`p.mul_(1 - 0.02*0.1)` per step, about e^-3 over 1525 st
    - severity: high
    - cls: interaction_cancels
    - link: L3
    - if_true: - metric: long_z
      - op: below
      - threshold: 1.0
  -
    - text: Fixing the residual growth does not change the loss.** Every consumer of the stream is behind an RMSNorm (`n1`, `n2`, final `norm`, plus `q_norm`/`k_norm`), so absolute stream scale only matters as a ratio between writers. The baseline grows MLP writes so that deep layers stay audible against the ac
    - severity: high
    - cls: below_noise
    - link: L4
    - if_true: - metric: long_z
      - op: below
      - threshold: 2.33
  -
    - text: Attention dominates the growth, so L2 fails.** Attention writes through `wo` and the Sinkhorn/`write_logits` combine are untouched. The 1.403→10.893 figure is the stream after each whole block, with the MLP share unmeasured. If attention writes carry most of the growth, the stream ratio stays at x7 
    - severity: medium
    - cls: interaction_cancels
    - link: L2
    - if_true: - metric: learn_z
      - op: below
      - threshold: 0.5
  -
    - text: Only helps early.** The x7.8 growth was measured at step 100, the high-LR transient. Over the 30% linear cooldown to lr=0 both arms settle, so any early edge from bounded writes shrinks by the end. The 40–45 step kill test cannot see this decay
    - severity: medium
    - cls: early_only
    - if_true: - metric: trend
      - op: below
      - threshold: 0.0
  -
    - text: Exam-time buffer handling.** The harness rebuilds `GPT(cfg)` and loads `model.pt`. If the EMA is registered with `persistent=False`, it loads at its init value (for example 1.0) instead of about 0.01, and every MLP write is off by about 75x, which is catastrophic. If the EMA update is not guarded by
    - severity: high
    - cls: impl_slip
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: Weight decay pulls the gain parameter toward the wrong value.** `a` is 0-d or 1-d, so `build_optimizers` puts it in the AdamW group with `weight_decay=0.1`. If `a` is initialized to log(initial write RMS), about log(0.013) ≈ -4.3 given the 0.02/√24 `w_down` init, decay pulls `a` toward 0, which is a
    - severity: medium
    - cls: impl_slip
    - if_true: - metric: learn_z
      - op: below
      - threshold: 0.0
  -
    - text: EMA lag is in micro-batches and its start is unspecified.** The momentum got truncated out of the spec ("mome…"), and `accum=8`, so the EMA updates 8 times per optimizer step. A momentum chosen "per step" lags 8x less than intended, and the batch-noise in c rises. An EMA initialized at 0 without bia
    - severity: medium
    - cls: impl_slip
    - link: L1
    - if_true: - metric: learn_z
      - op: below
      - threshold: 0.0
  -
    - text: In-place EMA update can break autograd.** If `ema` is used in `x/ema` (saved for backward) and then updated in place with `mul_/add_` in the same forward, backward raises a version-counter error. A builder who "fixes" this by computing c from the current batch's RMS before updating creates batch-dep
    - severity: low
    - cls: impl_slip
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: Feedback loop with a detached controller.** The gradient on the MLP matrices treats c as a constant, so the optimizer keeps pushing the output RMS up. τ steps later the EMA cancels it. Muon momentum (0.95) plus EMA lag form a second-order loop that can ring: the gain oscillates and so does the effec
    - severity: medium
    - cls: interaction_cancels
    - link: L1
    - if_true: - metric: learn_z
      - op: below
      - threshold: 0.0
  -
    - text: Global grad clip couples the MLP to everything else.** Gradients into the MLP matrices scale with c, which is ∝ 1/
    - severity: medium
    - cls: hidden_cost
    - if_true: - metric: long_z
      - op: below
      - threshold: 0.0
  -
    - text: Scale invariance plus weight decay raises the late effective LR.** With the output normalized, `w_up`/`w_down` behave like BatchNorm-plus-weight-decay weights: their norm shrinks, while Muon's fixed-size update stays the same, so the angular step grows over training. This partly cancels the cooldown
    - severity: low
    - cls: early_only
    - if_true: - metric: trend
      - op: below
      - threshold: 0.0
  -
    - text: Hidden host syncs.** Logging `exp(a)/ema` every 10 steps, or a builder writing `float(ema)` or `.item()` inside the forward, adds 12 layers × 8 micro-batches of host syncs per step. The 1% step-time prediction breaks, though it stays well under the 2x limit
    - severity: low
    - cls: hidden_cost
    - if_true: - metric: step_ratio
      - op: above
      - threshold: 1.05

premortem_verdict:
  Under pre-norm RMSNorm reads and Muon's fixed-size orthogonalized updates, the AGC only re-parameterizes write magnitude into `exp(a)` that the network can relearn, so the final exam loss stays inside seed noise (no 0.005-nat, p<0.01 win); the only way it moves the metric visibly is downward, throug

## Warmup-only AGC then freeze/fold  (picked_lane; part Per-token computation: SwiGLU MLP; outward; prior: new)
**Warmup-only AGC, folded into w_down** (port, card deepen, p_success None)
- mechanism: Run AGC only during the first 10% of steps, when relative updates are around 15% per step, then freeze c, multiply it into `w_down.weight`, and remove the scaling, leaving the exam architecture unchanged.
- prediction: Lower loss variance across seeds over steps 0–150; final loss at least equal to baseline.
- test: Two 200-step runs with 3 seeds each; compare per-step loss standard deviation over steps 0–150.
- source: own

Other sketches in this group:
- AGC only during warmup and early training (): Apply c only for steps 0–200, while norms grow fastest, then freeze c as a constant buffer, so the controller never fights the weight equilibrium (τ = 500).
- Warmup-only gain lock (port): Run the EMA gain only during the first 100 steps, the fastest growth phase. At step 100, freeze c and fold it into w_down. After that, nothing batch-dependent remains, and Muon trains the folded matrix as usual.

## Mixed activation populations  (picked_lane; part Per-token computation: SwiGLU MLP; outward; prior: new)
**AGC on the attention write (`wo`)** (port, card deepen, p_success None)
- mechanism: The same detached EMA gain on the attention branch output, targeting layer 0, where attention gradients are about 10x their neighbours' (z 7.0).
- prediction: The layer-0 attention gradient anomaly shrinks below 3x its neighbours' predicted value.
- test: 100-step run on 3 seeds; recompute the layer-0 gradient anomaly.
- source: own

Other sketches in this group:
- AGC on the attention write (): The same detached-EMA scalar gain on `wo`'s output, which tests whether the attention writes (quadratic in matrix scale) drive the late linear stream growth.

## Lower mixing logit init (less identity)  (picked_main; part State across depth: four-stream Sinkhorn residual mixing; inward; prior: None)

### Interrogation

chain:
  -
    - id: L1
    - text: Lowering the `mix_logits` init from 6 to 3, 2 or 1 makes more of each stream flow into the others at every mixing stage.
    - quantity: off-diagonal mass per stage, 3·b, read from the Sinkhorn output at step 0 and at the end
    - baseline: 0.74% at step 0; about 4.8% at the end from weight decay alone (belief)
    - expect: about 13% at init 3, 29% at init 2 and 52% at init 1 at step 0, growing over the run as decay shrinks the logits
    - breaks_if: init 6 already ends at 10% or more, or init 2 logits climb back above 4 (the optimiser re-sharpens to identity).
  -
    - id: L2
    - text: Read weights leave uniform, so the mixing matrix starts to affect the loss at all.
    - quantity: max abs(softmax(read_logits) − 0.25) over the 24 stages
    - baseline: 0 at init; unmeasured later
    - expect: above 0.05 by step 100, driven by Muon
    - breaks_if: it stays below 0.02 for the whole run, so the mixing has no effect on the loss at any init.
  -
    - id: L3
    - text: More exchange shrinks the differences between streams.
    - quantity: spread ratio ‖x − mean‖/‖mean‖ at block 11
    - baseline: unmeasured; it should be small because the write asymmetry is only 2%
    - expect: init 2 or lower gives a ratio at least 10x below init 6
    - breaks_if: the ratio is higher at init 2 than at init 6 (exchange creates differences instead of erasing them).
  -
    - id: L4
    - text: Smaller spread removes per-stream dominance but leaves the mean stream unchanged.
    - quantity: max-stream RMS / mean-stream RMS at block 11; mean-stream RMS growth
    - baseline: ×7.8 RMS growth in the single-stream baseline; this design is unmeasured
    - expect: the dominance ratio moves toward 1.0 at low init; mean-stream RMS growth changes by less than 3%
    - breaks_if: mean-stream RMS growth changes by more than 10% (column sums are not 1, or a precision artefact).
  -
    - id: L5
    - text: Less stream individuality changes the exam loss.
    - quantity: exam val loss over 3 seeds
    - baseline: pooled baseline mean (value not given)
    - expect: within about ±0.005 of init 6, drifting toward single-stream behaviour; a win larger than 0.005 is unlikely
    - breaks_if: init 2 or 3 beats init 6 by more than 0.005 at p < 0.01.

weakest:
  L2

qa:
  -
    - q: Does the loss depend on `mix_logits` at all while `read_logits` are uniform?
    - a: No. Sinkhorn ends with a column normalisation, so the column sums are exactly 1 in fp32 and the stream mean is invariant to M. Every read with uniform weights and the final `x.mean(dim=2)` read only the mean. So the forward pass is independent of M, and the gradient to `mix_logits` is zero up to bf16 rounding until the read logits move.
    - decides: rank the sweep by 45-step smoke loss vs rank it by a stream diagnostic or by full runs.
  -
    - q: Which way does a lower temperature move the differences between streams?
    - a: At init, M = aI + b(J − I), with eigenvalue 1 on the mean and a − b on every difference direction. - The per-stage factor a − b is 0.990 at init 6, 0.827 at init 3, 0.615 at init 2 and 0.30 at init 1. - Over 24 stages this compounds to 0.79, 0.010, 8.6e-6 and about 3e-13. - So inits of 2 or lower wipe out stream differences within a few blocks. The model then collapses to a single residual stream, which is the opposite of the stated mechanism.
    - decides: sweep {1, 2, 3, 6} vs sweep only {3, 4, 6} or switch to a non-contracting mix.
  -
    - q: Can streams 2 and 3 ever differ?
    - a: No (belief, from symmetry). - Both streams start as the same embedding copy and have identical write logits (−0.01 in both stages) and identical read logits. - The mix logits are symmetric under swapping 2 and 3, so the forward pass is equivariant under that swap. - Their gradients and updates are therefore identical, and they stay clones for the whole run. The design is effectively 3 streams, and no temperature fixes this.
    - decides: keep the current `write_logits` init vs give all four streams distinct read and write inits.
  -
    - q: Which optimiser updates the stream logits, and what does weight decay do to them?
    - a: Placement follows from the `ndim` filter in `build_optimizers`: - **`mix_logits` (2,4,4):** 3-D, so it goes to AdamW at 3e-3 with weight decay 0.1. Decay alone multiplies the diagonal by exp(−0.1·3e-3·~1281 effective steps) ≈ 0.68. That takes init 6 to about 4.1 (1.6% off-diagonal) and init 2 to about 1.36 (14.5% off-diagonal), so decay is already a hidden temperature schedule. - **`read_logits` and `write_logits` (2,4):** 2-D, so they go to Muon at lr 0.02 with weight decay 0.1. Newton–Schulz rescales the update to unit spectral norm, so even tiny gradients produce updates of about 0.01 per e
    - decides: leave the stream logits in the current groups vs a separate AdamW group with weight decay 0.
  -
    - q: Can a 45-step smoke run rank the inits?
    - a: No. - Warmup lasts 30 steps, so by step 45 the read logits have moved only about 0.2. The effect of M is then second order (read deviation × stream spread) and is swamped by rounding and seed noise. - C7 asks for a gap above 0.005 nats at p < 0.01, which only full runs can show. - A same-seed smoke sweep is useful only as a sanity check that the loss curves are nearly identical.
    - decides: smoke sweep vs going straight to full-length runs of two values.
  -
    - q: Does the 4-stream design beat the single-stream baseline at all?
    - a: Unknown; nothing in the brief measures it. At 1B scale, static ×4 hyper-connections gave a small gain over the baseline, and dynamic ×4 did better (https://arxiv.org/pdf/2409.19606). Because low temperature converges to single-stream behaviour (A2), a neutral 4-stream result implies a neutral sweep.
    - decides: spend runs on temperature vs first ablate to n = 1.
  -
    - q: Does the mixing push the residual through bf16?
    - a: Belief, not run. - Under autocast the einsum lowers to bmm, which runs in bf16. The carried residual is therefore rounded to bf16 at all 24 stages, even though the write addition promotes back to fp32. - After the bf16 cast, column sums are off by about 4–5e-4 at both init 6 and init 2. That shrinks the mean by roughly 1% over depth. Temperature barely changes this, but it adds noise to any comparison.
    - decides: compute the mix in fp32 with autocast disabled vs leave it as is.
  -
    - q: Can the final readout use stream differences?
    - a: No. `x.mean(dim=2)` is invariant to M and to how the signal is spread across streams. Only the 24 intermediate reads can exploit differences, which caps what any mixing temperature can buy.
    - decides: keep the mean readout vs a learned softmax final read.
  -
    - q: Where does the literature say Birkhoff-constrained mixing ends up, and has a lower init been tested?
    - a: Neither paper tested a lower init. - sHC reports that mHC matrices collapse toward identity and that non-negativity blocks subtractive mixing (https://arxiv.org/pdf/2603.20896). - The stream-collapse paper reports that the mix stays near identity while one stream dominates, and that a symmetry-breaking init helps (https://arxiv.org/pdf/2606.03483).
    - decides: lower init vs a symmetry-breaking init.
  -
    - q: Does the 20-iteration Sinkhorn cost matter for step time, and does the init change it?
    - a: The init does not change the cost. - Each optimizer step runs about 40 tiny kernels per stage × 24 stages × 8 micro-batches forward, plus the backward, although M depends only on the parameters. The wall-time share on the 4050 is unmeasured. - At init, M is a symmetric circulant matrix, so Sinkhorn is a no-op there.
    - decides: 20 iterations per forward vs an exact parameterisation or mixing at fewer stages.

known_tagged:
  -
    - text: `mix_logits` is a (2,4,4) Parameter initialised to 6·I. At init the diagonal is 0.9926 and each off-diagonal is 0.00246, not 99.75% identity as the variant states.
    - prov: script
  -
    - text: Diagonal and off-diagonal entries at the other inits: 0.870/0.043 at init 3, 0.711/0.096 at init 2, 0.475/0.175 at init 1.
    - prov: belief
  -
    - text: Sinkhorn ends with a column normalisation, so column sums are exactly 1 in fp32 and the stream mean is invariant to M.
    - prov: script
  -
    - text: All 4 streams start as the same copy of the embedding (`unsqueeze(2).expand`). The read logits are zeros, so every read is the mean, and the output is `x.mean(dim=2)`.
    - prov: script
  -
    - text: Write gains are 2·sigmoid(logit), which is 1.015 on the favoured stream and 0.995 elsewhere. The streams therefore differ by only about 2% of each write. Stage 0 favours stream 0, stage 1 favours stream 1, and streams 2 and 3 are never favoured.
    - prov: script
  -
    - text: `read_logits` and `write_logits` are 2-D, so they go to Muon at lr 0.02 with weight decay 0.1. `mix_logits` is 3-D, so it goes to AdamW at 3e-3 with weight decay 0.1.
    - prov: script
  -
    - text: The effective learning-rate sum over the run is about 1281 steps × lr: 1525 steps, 30 warmup, 458 cooldown.
    - prov: script
  -
    - text: Decoupled decay alone shrinks the diagonal mix logit by a factor of about 0.68 by the end, taking 6 to 4.1.
    - prov: belief
  -
    - text: The einsum runs in bf16 under autocast, so the carried residual is rounded to bf16 at every stage.
    - prov: belief
  -
    - text: The change adds 0 parameters and costs no extra memory or step time; C3 and C4 are untouched.
    - prov: script
  -
    - text: Single-stream baseline at step 100: RMS growth ×7.8, loss 6.1789, layer-0 attention gradient 0.24 against about 0.02 elsewhere.
    - prov: measured
  -
    - text: Static HC ×4 reaches an OLMo-1B eval loss of 2.791, against 2.781 for dynamic ×4. https://arxiv.org/pdf/2409.19606
    - prov: web
  -
    - text: On a 30M nanoGPT, alternative doubly-stochastic parameterisations reach only parity. https://arxiv.org/pdf/2604.02309
    - prov: web
  -
    - text: Version judged most likely to work: init 3 rather than 1 or 2, and only combined with distinct per-stream inits. On its own, expect a null result.
    - prov: belief
  -
    - text: What to log: spread ratio, max/mean stream RMS, read-weight deviation, the off-diagonal mass of M per stage at the end, and `torch.equal(x[:,:,2], x[:,:,3])`.
    - prov: belief

open:
  -
    - q: Does the 4-stream design beat n = 1 at all under this budget?
    - build: True
    - decides: temperature sweep vs dropping multi-stream
  -
    - q: How far do the read logits on Muon actually move by step 100 and by the end?
    - build: True
    - decides: leave them on muon vs move them to adamw with weight decay 0
  -
    - q: Does Newton–Schulz amplify noise-sized asymmetric gradients on the 2×4 write logits into real symmetry breaking?
    - build: True
    - decides: rely on the optimiser for symmetry breaking vs explicit distinct inits
  -
    - q: Is bf16 rounding of the residual at every stage costing loss relative to an fp32 mix?
    - build: True
    - decides: fp32 einsum under autocast disabled vs the current code
  -
    - q: What fraction of step time is the per-micro-batch Sinkhorn on the RTX 4050?
    - build: True
    - decides: 20 iterations vs an exact or sparse-stage mix
  -
    - q: What is the seed-to-seed spread of the exam loss for this script, and can 3 seeds resolve 0.005 nats?
    - build: False
    - decides: 3 vs 5 seeds
  -
    - q: Should weight decay on `mix_logits` be kept, since it already acts as a temperature schedule?
    - build: True
    - decides: weight decay 0.1 vs 0 on the stream logits

offshoots:
  - s32eadedb69
  - s2391ae38a8
  - sac31aaa55d
  - s76b0cb73ef
  - sf876630c85
  - s0942f451de

objections:
  -
    - text: The mechanism's premise is wrong. Every read already sees all four streams through `softmax(read_logits)` over dim 2, so differences created by the writes reach every later read with no exchange at all. A doubly-stochastic mix acts on the deviation subspace (x − mean) only as a contraction. Its eige
    - severity: high
    - cls: interaction_cancels
    - link: L3
    - if_true: - metric: long_z
      - op: below
      - threshold: 2.33
  -
    - text: The loss is blind to M while reads are near uniform. The final readout is `x.mean(dim=2)`, and the last Sinkhorn step normalises columns to exactly 1 in fp32, so the mean stream is invariant to mixing. With `read_logits = 0`, the gradient into `mix_logits` is zero up to bf16 rounding. The loss depen
    - severity: high
    - cls: below_noise
    - link: L2
    - if_true: - metric: learn_z
      - op: below
      - threshold: 1.0
  -
    - text: Weight decay moves every arm toward full averaging. AdamW decay of 0.1 × 3e-3 over about 1280 effective full-lr steps shrinks each logit by about 0.68x. Init 6 ends near 4.1 (4.8% off-diagonal), init 2 near 1.36 (about 45%), and init 1 near 0.68 (about 66%, close to J/4 flat averaging). The low-init
    - severity: medium
    - cls: early_only
    - link: L1
    - if_true: - metric: trend
      - op: below
      - threshold: 0.0
  -
    - text: The 45-step smoke sweep cannot rank the inits. `--smoke-steps 45` sets total_steps = 45, so warmup is 1 step and cooldown starts at step 31, which is a different schedule from the full run. By step 45, Muon has moved `read_logits` at most about 0.45, and the 2% write asymmetry has produced almost no
    - severity: medium
    - cls: early_only
    - if_true: - metric: learn_z
      - op: below
      - threshold: 1.0
  -
    - text: Implementation slip in parameter routing: `read_logits` and `write_logits` are 2-D (2×4), so `build_optimizers` sends them to Muon at lr 0.02, while `mix_logits` (3-D) goes to AdamW at 3e-3.
    - severity: medium
  -
    - text: Muon's Newton–Schulz step normalises noise into full-size updates of about 0.01 per entry per step. With decay, the logits equilibrate near ±5, so write gains `2*sigmoid` can swing anywhere in [0, 2].
    - severity: medium
  -
    - text: The orthogonalisation also forces the attention-stage and MLP-stage rows' updates to be mutually orthogonal, which is an arbitrary coupling.
    - severity: medium
  -
    - text: Read/write dynamics run about 3x faster than the mix logits and dominate both stream behaviour and seed variance, so the mix init is a second-order knob.
    - severity: medium
  -
    - text: Cheapest check: log `write_logits` and the read softmax at steps 100, 500 and 1500 across 3 seeds.
    - severity: medium
  -
    - severity: medium
    - cls: impl_slip
    - link: L2
    - if_true: - metric: long_z
      - op: below
      - threshold: 2.33
  -
    - text: The residual runs in bf16. Under autocast, `torch.einsum` dispatches to bmm in bf16, so from block 0's first `combine` onward the 4-stream residual is bf16; the baseline keeps an fp32 residual because fp32 x plus a bf16 branch promotes. At RMS of about 10 in late layers, updates below about 0.04 are
    - severity: medium
    - cls: impl_slip
    - link: L4
    - if_true: - metric: long_z
      - op: below
      - threshold: 0.0
  -
    - text: The target metric cannot move. Mean-stream RMS is invariant under column-stochastic mixing (L4 concedes this). Total RMS includes only the roughly 2% spread, so removing that spread changes the ×7.8 growth by less than 0.1%. The growth comes from the branch outputs and the mixing never touches it.
    - severity: low
    - cls: metric_blind
    - link: L4

premortem_verdict:
  On a doubly-stochastic, swap-symmetric mix, a lower init only contracts stream differences (about 0.6 survival per stage at init 2), so the model collapses to the single-stream baseline while still paying the bf16 residual and Sinkhorn overhead, and it lands within noise of the baseline or slightly 

## Leapfrog attention/MLP stream split  (picked_lane; part State across depth: four-stream Sinkhorn residual mixing; inward; prior: new)
**Leapfrog split of the four streams** (leap, card diagnosis, p_success 0.06)
- mechanism: Rerepresented as a Hamiltonian system, streams 0-1 are the position q and streams 2-3 the momentum p; attention reads p and writes q, the MLP reads q and writes p (read logits init +3 on its half, write gains 1.9 on its half and 0.05 on the other). Each branch then sees the embedding plus the accumulated outputs of the other kind only, each half receives 12 writes instead of 24, and the two halves stay bounded separately; the softmax reads and the Sinkhorn leak let the model reopen a direct attention-to-attention path if it needs it for induction.
- prediction: Residual RMS growth halves; recall-token loss worsens at first (induction composition is indirect) and must recover by step 300; loss at 50M tokens 0.005 nats lower or worse; parameters and time unchanged.
- test: 45 matched steps; then at step 300 compare recall-token loss to baseline, which decides the induction question directly.
- breaks: Every branch reads and writes the same averaged state; here attention and MLP each read the other's accumulated outputs.
- keeps: Four streams, static doubly-stochastic mixing, token-independent reads, mean at the head, same parameter count.
- kill: Loss at step 45 above 6.78, or recall-token loss above baseline by more than 0.1 nats at step 100.
- target: residual RMS growth, down; recall-token loss (must not rise by more than 0.05 at matched steps)
- scope: validation loss at 1024
- source: own

## Unit-scale embedding with small final gain  (picked_main; part State across depth: normalization placement and initializati; outward; prior: new)
**Unit-scale embedding, temperature in the final gain** (tune, card research, p_success None)
- mechanism: Initialize the tied embedding at std 1.0 instead of 0.02 and the final norm gain at 0.05 instead of 1, so initial logits keep std near 1. This inverts assumption 3, uniform std 0.02 regardless of role. The embedding is under Adam at lr 3e-3 and moves slowly, while branch writes grow under Muon, so by mid-training the token identity is a few percent of the stream for every layer past 0. At std 1 the identity signal stays comparable to the writes throughout the budget, which is what Inkling's embedding norm and the speedrun's large embedding learning rate also buy.
- prediction: No parameter or step-time change. Initial loss within 0.2 of the baseline's. Faster descent in the first 200 steps and lower loss on rare tokens, whose identity otherwise fades deepest in the stack.
- test: Sweep embedding std over 0.3 and 1.0 and final gain init over 0.02, 0.05 and 0.1 in 45-step matched-seed runs. Check the first-step loss is under 11.5.
- target: loss on rare tokens: down
- scope: validation loss at 1024
- source: own

## Progressive stacking (growth schedule)  (picked_main; part State across depth: layer count and block layout; inward; prior: new)
**Progressive stacking from six to twelve blocks** (port, card research, p_success None)
- mechanism: For the first quarter of the steps only blocks 0 to 5 run; at the growth step the model copies each of them into block 6 to 11 and the loop switches to all twelve. The shallow phase learns the unigram and bigram structure at 0.7 times the step cost and a steeper per-token slope, and the copied stack starts from a trained function rather than a cold one, which the stacking literature reports as 1.5 to 2 times token efficiency.
- prediction: Parameter count unchanged; first-phase steps about 0.7 times the baseline's time, full-phase unchanged; a loss bump of under 0.1 nats at the growth step that recovers within 50 steps, and a curve 0.02 to 0.04 nats below the baseline by the start of cooldown.
- test: Run with growth at step 100 for a 300-step budget and compare the matched-step curve; if the post-growth loss is still above the baseline at step 300 the idea dies. The growth step fraction is the one constant to sweep, over 0.15, 0.25 and 0.4.
- target: validation loss only
- scope: validation loss at 1024
- source: https://arxiv.org/abs/2405.15319

## Partial untie / private output parameters  (picked_lane; part Inputs and outputs: tied embedding, stream read-out, logits; inward; prior: new)
**Low-rank untied correction fused into the head** (port, card research, p_success None)
- mechanism: Head weight becomes the shared table plus A times B with rank 32 (A is 50304x32, B is 32x384, B zero-init), computed as one fused matmul over the concatenated features so no second full logits tensor is allocated. This gives the output side its own cheap degrees of freedom, letting the shared rows move back toward input-shaped geometry, and probes the open question of a low-rank untie at a budget where full untying is forbidden. At scale this is the affordable direction, since full untying grows with vocabulary while the correction grows with rank.
- prediction: Parameters +1,622,016 to about 22.86M, under the 23.37M cap; step time +3 to 4 percent; peak memory +50 MB for the rank-32 factor and its optimizer state; loss gain, if any, appears after the first thousand steps as B grows from zero.
- test: Build-only count, 3 steps for time and memory (the fused form is the pass or fail here), 45 matched steps for an early read, then a 300-step check since the effect is late.
- target: loss on rare tokens, down
- scope: validation loss at 1024
- source: https://github.com/KellerJordan/modded-nanogpt and https://arxiv.org/pdf/2603.26663

Other sketches in this group:
- Partial untie, 16 output-only dimensions (leap): Drop "all rows share one table". The head uses `cat(E[:, :368], U)` with U of size 50304×16 as an output-only parameter, while the input still reads all 384 dimensions of E. This gives a small private channel to the output side for frequency and anisotropy.
- Partial tie with a private output subspace (leap): Drop "the classifier is the whole input row". The head uses [E_v ; P_v], where P is a private 50304×32 output-only block. P is fed by a 384→32 projection of the normed read-out (about 1.6M parameters, so about 22.85M, under the 23.36M cap). Output-shaped features (frequency, next-token-class) can li
- Frequency-split tying (untie only the frequent rows) (leap): The brief says the tied matrix ends up output-shaped because output gradients dominate, and that rare rows (loss 11.6 nats at rank 10k+, which equals the unigram cost) only learn frequency, so they are the rows tying protects. Setting "the two sides conflict most where both gradients are large" as a

## Rescale embedding input to match block write scale  (picked_main; part Inputs and outputs: tied embedding, stream read-out, logits; inward; prior: new)
**Normalize the row before it enters the stream** (port, card diagnosis, p_success None)
- mechanism: Anomaly row restated: block-0 MLP gradient is 0.0718 above what layers 1 and 2 predict on all three seeds. Rival causes: (a) the embedding enters at RMS 0.02 while block 0 writes at RMS 1.4, so block 0 is the real embedding and MLP-0 is a bigram table; (b) a training-phase effect, at step 100 only shallow layers learn and the excess vanishes by step 500; (c) a measurement artifact, raw gradient norm under pre-norm scales as one over the residual RMS at that depth and layer 0 sits where RMS is smallest, so the neighbour extrapolation is wrong. Five-minute probe on the step-100 checkpoint: multi
- prediction: Block-0 attention and MLP gradient norms fall toward the layer 1-2 trend; the residual RMS after block 0 rises from 1.4 to about 2; the loss curve is ahead by 0.05 to 0.10 nats at step 100 and val loss ends 0.008 to 0.015 lower. Params plus 384, step time unchanged.
- test: 100-step smoke on seed 0 against the recorded curve; kill if step-100 loss is not at least 0.03 below 6.179 and the layer-0 excess has not halved. Fallback sweep in TEST of the tune below.
- target: gradient norm spread, down (layer 0 toward trend); residual RMS growth, down
- scope: validation loss at 1024
- source: https://www.sebastianraschka.com/blog/2026/inkling-architecture-benchmark-notes.html (embedding RMSNorm)

Other sketches in this group:
- Fixed input scale on the tied row (tune): The same cause as above with no norm: multiply the looked-up row by a fixed constant before the broadcast so the input is not 70x smaller than block 0's write. Keeps row norm visible to the input, which the norm port removes, so the two together tell whether row norm carries useful frequency signal 
- Embedding RMSNorm (input scale decoupled from the tied head) (port): Add an RMSNorm with learned gain directly after the token lookup, so every token enters the four streams at unit RMS. With tied embeddings the row norms of E are set by the head gradient and AdamW decay, so rare tokens enter at a fraction of the scale of frequent ones and are drowned by the first br
- Embedding output multiplier √d (unit-RMS stream entry) (port): Multiply `tok(idx)` by √384 ≈ 19.6 before expanding to 4 streams, so the stream starts at RMS about 0.4 instead of 0.02. This targets the 10x layer-0 gradient anomaly through the norm Jacobian (∝ 1/RMS).
- Input-embedding scalar to fix the tiny layer-0 stream (port): Put one learned scalar (no decay, init about 10) on the token embedding before it is expanded into the 4 streams, so the raw stream RMS is not about 0.02 against a 1.4–10.9 residual.

## Gated expansion of K/V  (picked_lane; part Sequence mixing: latent KV compression and expansion; inward; prior: new)
**Gated nonlinear expansion** (leap, card diagnosis, p_success 0.15)
- mechanism: Linearity of expand is inherited from the inference recipe, but here the 128-d latent is a true bottleneck that six heads must share. Make the expansion a gated map, expand(c) times silu(gate(c)), so keys and values become a two-layer function of the latent and the six heads can carve different nonlinear views from the same 128 channels.
- prediction: validation loss lower by 0.01; parameters up 98,304 per layer, 1.18M total; step time up under 2%.
- test: build-only count, 3-step timing, 100-step matched run; killed if the gate's mean activation stays near 0.5 and loss is not below baseline by step 100.
- breaks: keys and values are a linear image of the latent.
- keeps: compression, the norm, same latent width, same q path, same scores.
- kill: at 45 matched steps, loss not below baseline, or gate saturation above 90% of units.
- target: validation loss only; secondary attention entropy in deep layers, down.
- scope: validation loss at 1024
- source: own

Other sketches in this group:
- Separate K and V gates (): Drop "K and V share one gated code". Keep one ungated linear latent for K, and gate only the V half of the expansion with silu(W_g c), so retrieval geometry stays linear while the content read out is selected per token.

## QK-norm / init scale for rotary slices  (picked_main; part Sequence mixing: decoupled rotary dims and QK-norm; inward; prior: new)
**QK-norm over the rotary channels too** (port, card research, p_success None)
- mechanism: q_norm and k_norm cover only the 64 content dims, so the 32 rotary dims enter the 1/sqrt(96) softmax unnormalized: at init their logit std is about 0.09 against 0.82 for content (a 10x gap), and later they are unbounded, so the 22.7 max logits can be carried by the position term. Add an RMSNorm with its own gain on the per-head rotary query and on the shared rotary key, so position and content compete at a controlled scale from step 0, which DeepSeek-V4 does by norming whole query heads.
- prediction: previous-token and induction heads form earlier, so the loss at step 100 is lower by about 0.03 to 0.05; final loss lower by 0.01 to 0.02; parameters +64 per layer; step time unchanged.
- test: 45 matched steps against baseline, logging the std of the rotary-only and content-only logit contributions at steps 0 and 45; if the curve is not below baseline by step 45, drop it.
- target: max attention logit, down
- scope: validation loss at 1024
- source: https://arxiv.org/html/2606.19348v1

Other sketches in this group:
- QK-norm extended to the rotary slices with learned gains (port): Anomaly row: layer 2's attention gradient (0.0259) is 0.0039 below what layers 1 and 3 predict, on three seeds. Rivals: (a) layer 2 is partly idle, its heads near bag-of-words (entropy 0.85) and redundant with layers 1 and 3; (b) measurement artifact: the neighbour predictor is linear, the attention
- Rotary projection init scale (tune): The rotary projection and the rotary slice of wq start at std 0.02 like every other matrix, but their logits are a dot product of two tiny unnormalized vectors, so Muon's fixed-magnitude updates need many steps before position matters at all. Multiplying the init of these two weight slices by 3 brin
- Normalized rope slice (port): qr and kr are the only unnormalized pieces of the score, so the maximum logit of 22.75 at layer 8 can come from them growing without a bound while the content slice is pinned by q-norm and k-norm. RMS-normalize qr per head and kr once, with learned gains, so the whole dot product has a controlled sc
- Normalize the rotary logits too (RoPE-part QK-norm with its own scalar) (leap): Apply a no-affine RMSNorm to qr and kr before RoPE, with one scalar per layer. Both parts of the logit then have explicit temperatures, and the unbounded rotary logit that likely drives the max logit of 22.75 is capped.

## Relative (stream-ratio) AGC  (picked_lane; part Per-token computation: SwiGLU MLP; outward; prior: new)
**Stream-relative write ratio** (leap, card deepen, p_success None)
- mechanism: Drop "the target is a fixed per-layer level". Set c = exp(a)·sg(rms(stream input))/ema_rms(write), so `a` sets the write-to-stream ratio (init a = log 0.5, matching step 0) and the write follows stream growth automatically.
- prediction: The write/stream ratio stays within 2x of exp(a); stream growth is still possible but compounds through attention and embedding only.
- test: 45 matched steps; log the ratio per layer and the loss over steps 40–45.
- breaks: the write magnitude is set in absolute units by the branch's weights
- kill: mean loss over steps 40–45 more than 0.03 nats above baseline
- source: own

Other sketches in this group:
- Relative AGC (write-to-stream ratio control) (leap): Drops "absolute write RMS matters". Scale the MLP write to exp(a)·ema(block-input stream RMS)/ema(write RMS), so each layer contributes a learned, fixed fraction of the stream it writes into, whatever the stream's size.

## Symmetry-breaking / distinct write inits  (picked_main; part State across depth: four-stream Sinkhorn residual mixing; inward; prior: known)
**One-hot symmetry-breaking write init** (port, card research, p_success None)
- mechanism: The current write init gives gains of 1.015 versus 0.995 across streams, which is symmetric to within 2%, so the streams remain near-copies and the mixing has nothing to mix; Hyper-Connections initialize each stage's write to a single stream, and the stream-collapse paper shows symmetry-breaking init reduces one-stream dominance. Initialize write logits so stage s of block ℓ writes with gain 1 to stream (2ℓ+s) mod 4 and gain about 0.02 to the others (logit -4.6).
- prediction: No parameter or time change; per-stream RMS becomes balanced (four streams of similar norm instead of one dominant); loss curve matches baseline early and ends 0.005 to 0.01 lower.
- test: 45-step matched smoke run plus the stream RMS printout; the full 3-seed run decides.
- target: residual RMS growth, down
- scope: validation loss at 1024
- source: https://arxiv.org/pdf/2409.19606 and https://arxiv.org/pdf/2606.03483

Other sketches in this group:
- Distinct stream inits (port): Give all four streams distinct write-logit patterns (stream = stage index mod 4 favoured, ±0.5) and slightly distinct read logits. This removes the exact 2↔3 clone and gives the mixing real differences to act on.
- Fixed-sum rotating writes (port): Parameterise writes as 4·softmax(write_logits) so the total write into the mean stays exactly 4, and init the favoured logit at +2, rotating which stream is favoured with (2·layer+stage) mod 4. Streams then differ by O(1) instead of 2%, and all four get used.
- Staggered writes with staggered reads (from the layer-2 gradient dip) (port): Restated row: block 2's attention gradient norm sits 15% below the line through blocks 1 and 3, on all three seeds. Rival causes: (a) block 2's attention is partly redundant because blocks 1 and 2 read the same uniform stream mix one stage apart and the four streams are near-identical copies (writes
- Multi-timescale streams (subtract the identical copies, reinvest in graded write rates) (leap): Removal: the near-symmetric write start, which keeps all four streams within 2% of each other so three of them are copies that cost memory and buy nothing. Reinvestment: initialize the write logits so the streams receive each branch at weights 1.6, 0.8, 0.4, 0.2, making stream 3 a slow near-embeddin

## Hard one-hot reads  (picked_lane; part State across depth: four-stream Sinkhorn residual mixing; inward; prior: new)
**Hard one-hot reads, mixing as the only exchange** (leap, card deepen, p_success None)
- mechanism: Remove read_logits. Attention at stage s reads stream (2·layer + 0) mod 4, the MLP reads (2·layer + 1) mod 4, and the output stays the mean. Streams become distinct paths, and the Sinkhorn matrix at init 2 is the only cross-stream route, so mixing must carry real information.
- prediction: Stream spread rises by about 10× over the soft-read design and the mix logits move measurably away from init. Loss is either worse early or better late.
- test: Run a 200-step smoke against the init-6 soft-read run at matched steps.
- breaks: Each branch reads a learned soft combination of all streams.
- kill: Mean training loss over steps 90–110 more than 0.05 nats above the init-6 four-stream run over the same steps.
- source: own


# The interrogated ideas (all eight)

## Short causal conv on keys/values  (part Sequence mixing: head count, head structure and output proje; prior: None)

### Chain and facts

chain:
  -
    - id: L1
    - text: Delta-parameterised depthwise causal taps (lags 0–3) on the expanded `kv` tensor -> keys and values in some layers pick up previous-token content
    - quantity: per-layer
    - baseline: 0 (identity init)
    - expect: up; mean
    - breaks_if: every lag≥1 tap stays below 0.05 at step 300, or the taps shrink back toward 0 after rising
  -
    - id: L2
    - text: Previous-token-carrying keys -> one head in an early layer attends to j where x_{j-1}=x_t (a single-layer induction pattern)
    - quantity: attention entropy in layers 0–3; induction score (attention mass on "token after the previous occurrence of x_t")
    - baseline: entropy L0 0.957, L1 0.903, L2 0.854, L3 0.804; induction score unmeasured
    - expect: entropy down 0.05–0.2 in at least one of layers 0–3; induction score up at least 2x vs baseline at matched step
    - breaks_if: entropy and induction score match baseline within seed spread at step 300
  -
    - id: L3
    - text: Induction pattern -> recall-token loss falls faster
    - quantity: recall-token loss (13% of targets)
    - baseline: 4.475 at step 100 (rest 6.704)
    - expect: below baseline by 0.05–0.3 nats at matched steps from step ~200–300 onward
    - breaks_if: gap is within seed noise at step 1000
  -
    - id: L4
    - text: Copying from far context -> late positions gain more than early ones
    - quantity: loss(256–1024) minus loss(0–64)
    - baseline: +0.459 at step 100
    - expect: smaller by 0.05–0.2 at matched step
    - breaks_if: unchanged or larger
  -
    - id: L5
    - text: Recall gain plus phrase-level key/value matching, with non-recall tokens unharmed -> exam val loss down
    - quantity: exam val_loss after 1525 steps
    - baseline: pooled mean unmeasured here
    - expect: down 0.01–0.04 nats
    - breaks_if: non-recall loss rises enough to cancel, or the net gain is under 0.005 or not significant at 3 seeds

weakest:
  L2

qa:
  -
    - q: Does the 2.2-nat recall gap measure induction, or just that bigrams repeated within a window are frequent and easy?
    - a: Unknown. The measurement does not control for frequency, and repeated bigrams skew toward common ones, so part of the 4.475 vs 6.704 gap is unigram/bigram statistics. Build the controlled metric. Without it, a move in L3 cannot be attributed to L2.
    - decides: target metric "recall-token loss" vs a frequency-controlled metric (second vs first in-window occurrence of rank>1000 bigrams)
  -
    - q: Does the baseline form an induction head inside the 1525-step budget anyway?
    - a: Unknown for this setup. Olsson et al. saw the phase change at roughly 2.5–5B tokens in their models (https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). Muon-trained small models learn far faster, so that number does not transfer. Layer-0 entropy of 0.957 at step 100 shows no sharp positional head yet. If the baseline forms one by step ~300, the conv only buys a head start, and the step-1525 gap may shrink below 0.005.
    - decides: taps in all 12 layers vs only early layers; also whether the end-of-run gain can persist or only the curve shifts earlier
  -
    - q: Which optimizer group and which parameterisation should the taps get?
    - a: `build_optimizers` sends every 2-D parameter not named tok/head to Muon. A (4,768) tap tensor would get orthogonalised updates scaled by sqrt(768/4)≈13.9 at lr 0.02, which wrecks it. In AdamW, wd 0.1 at lr 3e-3 pulls a raw identity tap down by about e^-0.38≈0.68 over the run when no gradient opposes it. Use the delta form, route it to AdamW explicitly or store it 3-D, and preferably give it wd 0.
    - decides: raw kernel with identity tap vs delta form (kv + Σ w_l·shift(kv,l), w init 0); Muon vs AdamW
  -
    - q: Should the taps go before or after `k_norm`, and do they touch the rotary key?
    - a: Put them before, on the contiguous `kv = expand(...)` of shape (B,T,768). k_norm then re-normalises the mixed key, so QK-norm still holds logit scale. A conv after k_norm would change key RMS and the effective 1/sqrt(96) temperature. Leave `kr` (shared 32-dim rope key) and `qr` untouched so position encoding stays per-token.
    - decides: conv on `kv` before view/k_norm vs on k after k_norm
  -
    - q: Should the taps sit on the 768 expanded channels or on the 128-dim latent?
    - a: Expanded gives independent kernels for key and value channels, per head, so a head can have prev-token keys and current-token values at once. That is the exact induction wiring. The latent version shares each kernel between every k and v channel that reads that latent dim and relies on `expand` to route. Primer puts its depthwise conv after the per-head projections (https://arxiv.org/abs/2109.08668). Both are far under C3, so pick expanded.
    - decides: expanded (36,864 params) vs latent before latent_norm (6,144 params)
  -
    - q: Should it be implemented with `F.conv1d` or with shift-and-add?
    - a: Shift-add on the contiguous (B,T,768) tensor avoids two transposes and copies, and costs three fused-ish elementwise ops per layer. Cast the taps with `.to(kv.dtype)` like the script does for the mixing weights, otherwise fp32 upcasts double the memory of those tensors. The step-time cost is unmeasured. Belief: 1–3% given that the 4-stream mixing already does similar elementwise work.
    - decides: conv1d with (B,C,T) transposes vs F.pad shift-add on (B,T,C)
  -
    - q: Can a recall-only gain pass C7?
    - a: Overall Δ ≈ 0.13·Δrecall + 0.87·Δrest, so recall tokens alone must improve by ≥0.04 nats just to clear the 0.005 margin. p<0.01 at 3 seeds needs more than that, and seed std is unmeasured. Primer reports MDHA gains in overall LM loss, not only on repeats (https://arxiv.org/abs/2109.08668). L5 therefore depends on phrase-level effects on non-recall tokens too.
    - decides: kv-only (recall-targeted) vs a broader q/k/v version aimed at all tokens
  -
    - q: Should the taps also go on queries?
    - a: Induction needs only shifted keys. Primer convolves Q, K and V with kernel 3 (https://arxiv.org/abs/2109.08668). If q is added, apply it only to the 64 content dims of the 576-wide `wq` output, never to the 32 rope dims. Start with kv-only for clean attribution.
    - decides: kv-only vs q-content + kv
  -
    - q: Can the conv leak future tokens into the exam?
    - a: The shift must be `F.pad(kv, (0,0,l,0))[:, :T]`, i.e. position t reads t−l. A reversed shift leaks x_{t+1}, gives a spuriously low loss, and the harness exam would reward it. The builder must check that changing idx[:, t+1] leaves logits[:, :t+1] unchanged. Positions 0–2 see zero padding in both training and exam windows, so the two are consistent.
    - decides: left pad (lags 0..3) vs any centred/right pad; whether a causality test is mandatory
  -
    - q: Does the 45-step/1000-step test plan match the real schedule?
    - a: `lr_mult` uses `total_steps = smoke_steps`, so a 1000-step smoke run has a different schedule from the 1525-step run (warmup 30, cooldown from 1067). Both arms must use the same steps setting. Recall-loss checkpoints are not logged by the script, so they need a separate probe. KNOWN - `kv = expand(latent_norm(compress(x)))` is (B,T,768) and is viewed as (B,T,6,128) per head, then chunked into k(64) and v(64). A per-channel conv on `kv` before the view is equivalent to per-head k/v convs [script] - q and k are 96 wide (64 content + 32 decoupled rope, one shared rope key head); SDPA default scal
    - decides: `--smoke-steps 1000` (schedule compressed, cooldown from step 700) vs the full-run schedule truncated at 1000

### New ideas born here
  - sc12466a2ad
  - s30726e7798
  - s4174743845
  - sfb757e6226
  - s2da7836045
  - s99bc12e1f0

### Pre-mortem objections
  -
    - text: Conv on V corrupts the copied token.** An induction head attends to position j, the token after the earlier occurrence, and copies v_j. With the conv on v, v_j becomes a mix of v_j, v_{j-1}, v_{j-2} and v_{j-3}, so the head copies a smear of the matched context instead of the next token. The gain on
    - severity: high
    - cls: interaction_cancels
    - link: L3
    - if_true: - metric: learn_z
      - op: below
      - threshold: 1.0
  -
    - text: Conv weight routed to Muon.** If the builder stores the taps as a 2-D (768,4) or (4,768) parameter, `build_optimizers` sends them to Muon. Newton-Schulz then orthogonalizes a 768x4 matrix, and the shape scale `max(1, 768/4)**0.5 ≈ 13.9` turns lr 0.02 into about 0.28 per step on taps of magnitude abo
    - severity: high
    - cls: impl_slip
    - if_true: - metric: trend
      - op: above
      - threshold: 0.0
  -
    - text: Effect is early-only.** The baseline has 12 layers, which is plenty to build a previous-token head plus an induction head within 1525 steps (50M tokens). Recall tokens are already easy at step 100 (4.475 vs 6.704). The conv only shortcuts the formation, so the step-300 gap the chain predicts shrinks
    - severity: high
    - cls: early_only
    - link: L3
    - if_true: - metric: trend
      - op: below
      - threshold: 0.0
  -
    - text: Gain is too small for C7.** Recall tokens are 13% of targets. Even the chain's optimistic 0.05 nats on them, held to the end, is 0.0065 nats overall. That barely clears the 0.005 margin, and with 3 seeds a one-sided Welch test at p<0.01 needs the effect to be about 3x the seed std. Seed std on this 
    - severity: high
    - cls: below_noise
    - link: L5
    - if_true: - metric: learn_z
      - op: below
      - threshold: 2.0
  -
    - text: Conv after QK-norm, or over the RoPE half.** "Right before SDPA" means k is already `cat([k_norm(k), rope(kr)])`. Convolving there has two problems:
    - severity: medium
  -
    - text: It un-normalizes the content keys. Summed taps change logit scale, which interacts with the max logits of 15–22 in layers 3–10.
    - severity: medium
  -
    - text: It mixes RoPE-rotated components from positions t-1 to t-3 into position t. That breaks the relative-position property the 32-dim shared `kr` exists to provide, and position-based heads degrade.
    - severity: medium
  -
    - text: The conv belongs on the content k/v from `expand`, before `k_norm`, and never on `kr`. Cheapest check: look at where the conv sits in the code, then at the max attention logit per layer at step 300 against the baseline.
    - severity: medium
    - cls: impl_slip
    - if_true: - metric: fidelity
      - op: below
      - threshold: 0.5
  -
    - text: Non-recall tokens pay.** Layer 0 has near-flat entropy (0.957) and takes 10x the attention gradient of the other layers. The conv gives layer 0 a cheap local n-gram smoother, and its large gradient will pull the taps toward local averaging that serves bigram statistics rather than induction. Smeared
    - severity: medium
    - cls: interaction_cancels
    - link: L5
    - if_true: - metric: long_z
      - op: below
      - threshold: 0.0
  -
    - text: Redundant with the 4-stream mixing.** The static Sinkhorn residual streams already let early layers write previous-token features into separate streams, which later reads can pick up. The marginal value of a kv conv is lower here than in a plain residual network, so the taps stay near identity (the 
    - severity: medium
    - cls: interaction_cancels
    - link: L1
    - if_true: - metric: learn_z
      - op: below
      - threshold: 1.0
  -
    - text: Hidden VRAM and step cost.**
    - severity: medium
  -
    - text: Memory: `F.conv1d` needs a (B,C,T) layout, so there is a transpose/contiguous copy plus left padding. That saves roughly 3 extra (4,1024,768) bf16 activations per layer, about 0.2–0.3 GB over 12 layers. The baseline already sits close to the limit (micro_batch 8 spilled past 6 GB), so this can cross
    - severity: medium

Pre-mortem verdict: The conv shortcuts an induction circuit that the 12-layer baseline builds on its own within the first few hundred steps, and the conv on V blurs exactly the token being copied. Any recall gain on 13% of tokens fades by the end of the cooldown to well under 0.005 nats overall, inside the noise of 3 s

## Rotary values with un-rotated readout  (part Sequence mixing: decoupled rotary dims and QK-norm; prior: new)
**Causal key depletion (overgrazing)** (leap, card research, p_success 0.1)
- mechanism: Shared structure: in a consumer-resource system consumers arriving in order deplete a shared resource, so a patch grazed by many earlier consumers yields less to later ones; here queries are consumers arriving in position order, keys are patches, and P[i,j] is consumption. Instantiation, in the last 2 layers only: S = q·k/sqrt(96), P0 = causal softmax(S), D = exclusive cumsum of P0 over queries, S' = S − λ_h·log1p(D), output = softmax(S')·v, with λ_h per head initialised at 0 so training starts exactly at baseline. Where it breaks: ecological resources regenerate and depletion is physical, whi
- prediction: peak VRAM rises by roughly 0.1 to 0.3 GB under activation checkpointing of those two attention cores; step time up 10 to 25 percent; deep-layer entropy up; loss lower by 0.005 to 0.015.
- test: first a 3-step smoke for peak VRAM and step time (must stay under 5.6 GB and 9.5 s), then 45 matched steps.
- breaks: a key's attractiveness is independent of how much attention earlier queries already took from it.
- keeps: softmax over keys, causal mask, QK-norm, RoPE, all other 10 layers unchanged.
- kill: VRAM over 5.6 GB at the 3-step check, or loss at 45 steps above baseline by more than 0.02.
- target: attention entropy in deep layers, up
- scope: validation loss at 1024
- source: own

### Chain and facts

chain:
  -
    - id: L1
    - text: Rotate v[..., rotated slice] by R(ω·j) before SDPA, and rotate the same slice of the SDPA output by R(−ω·i) before `wo` -> each head's rotated slice of the output equals Σ_j a_ij R(ω(j−i)) v_j
    - quantity: unit check with forced one-hot attention at offset δ: the output should equal R(ωδ)v_j, and at δ=0 it should equal v exactly
    - baseline: n/a (identity today)
    - expect: max abs error under 2e-2 relative in bf16, and exactly 0 for the unrotated slice
    - breaks_if: the error exceeds 5e-2 at T=1024, or the δ=0 output is not v (sign or pairing bug)
  -
    - id: L2
    - text: For diffuse heads, the rotated slice cancels: Σ a_ij e^{iω(j−i)} shrinks for fast ω, so content in those dims is lost while offset information appears
    - quantity: per-layer ratio RMS(rotated slice of y) / RMS(unrotated slice of y)
    - baseline: unmeasured (≈1 by symmetry today)
    - expect: 0.2–0.7 in layers 0–2 (entropy 0.85–0.96), 0.6–1.0 in layers 6–10 (entropy 0.67–0.71)
    - breaks_if: the ratio is ≥0.95 in layer 0, which would mean no cancellation (and no offset signal) or a bug; or ≤0.3 in layers 6–11, which would mean the payload is destroy
  -
    - id: L3
    - text: The offset phase reaches the residual stream in a usable form -> a downstream probe can read the attended offset
    - quantity: linear probe from the block output to the argmax-attention offset (log-binned)
    - baseline: unmeasured
    - expect: probe accuracy up over baseline in layers ≥4, by any clear margin
    - breaks_if: the probe is no better than baseline (the phase is entangled with content and `wo`/MLP can't decode it)
  -
    - id: L4
    - text: Offset-aware reads help copying -> recall-token loss drops
    - quantity: recall-token loss (bigram already seen in window, 13% of targets)
    - baseline: 4.475 at step 100
    - expect: down 0.02–0.10 at step 100 (the idea's 4.30 target, −0.175, is optimistic)
    - breaks_if: the change is within seed noise, or recall loss rises
  -
    - id: L5
    - text: The cost of L2 stays small elsewhere -> non-recall loss is unchanged
    - quantity: non-recall loss
    - baseline: 6.704 at step 100
    - expect: within ±0.02
    - breaks_if: it rises more than 0.02 (lost value capacity outweighs offset info)
  -
    - id: L6
    - text: Net effect at the full budget
    - quantity: exam val loss at 50M tokens
    - baseline: pooled baseline mean (not in brief)
    - expect: −0.015 to +0.01
    - breaks_if: not below baseline by >0.005 at p<0.01 over 3 seeds

weakest:
  L4

qa:
  -
    - q: Does next-token prediction on recall tokens actually need the copy offset, or is it already implicit in which head did the copying?
    - a: In the standard induction circuit, the previous-token head always attends at offset 1, so its offset is implied by the head's identity. The induction head's prediction (the token after the earlier match) does not depend on how far back the match was. So the offset is mostly redundant for plain recall, and L4's gain probably comes from variable-offset heads, if it comes at all. This is a belief; it is unmeasured here.
    - decides: rotate every layer vs only layers whose heads attend at variable offsets (≥ mid-depth)
  -
    - q: Which frequencies should rotate the value: all 16 of the existing cache (32 dims) or a band?
    - a: `rope_cache(T, 32, 10000)` gives wavelengths 2π·10000^(k/16) for k = 0..15, from 6.3 up to about 35,300 tokens. - For k ≥ 13, the angle over the whole 1023-token window is under 0.6 rad, so those dims carry almost no offset information. - For k ≤ 2 (wavelengths 6–20), the phase aliases, and these dims cancel hardest under diffuse attention. - k = 3..10 covers wavelengths of about 35–1990 tokens, which is the "32 to 2048" range the first variant actually states. Note that variant's label is wrong: its "longest 8 wavelengths" would be k = 8..15, not 32–2048. I recommend the 16-dim band k = 3..10
    - decides: 32 dims with `cos, sin` as-is vs 16 dims using `cos[:, 3:11]`, `sin[:, 3:11]`
  -
    - q: How bad is the content loss from cancellation in the shallow, near-uniform layers?
    - a: With near-uniform attention over n tokens, |Σ a_ij e^{iωδ}| ≈ |sin(ωn/2)| / (n·sin(ω/2)), which is far below 1 except for the slowest ω. Layer 0 has entropy 0.957, so its rotated slice will carry close to zero content. Its attention gradient norm is already anomalous (0.24 against about 0.02 in its neighbours, z = 7 over 3 seeds), so perturbing layer 0's value path is the riskiest place. That last point is a belief.
    - decides: apply in all 12 layers vs skip layers 0–2
  -
    - q: Is the inverse rotation exactly what `apply_rope` can do, and is the pairing right?
    - a: `apply_rope` pairs dim i with dim i+D/2 inside the tensor it is given, and slices `cos[:T]`. So these calls give the exact inverse at query position i: - `apply_rope(v[..., :32], cos, sin)` for the forward rotation - `apply_rope(y[..., :32], cos, -sin)` for the inverse For the 16-dim band, call it on a 16-dim slice with `cos[:, 3:11]` and `sin[:, 3:11]` (8 pairs). Apply it to `y` while `y` is still (B, H, T, 64), before the transpose. This is read from the script.
    - decides: reuse `apply_rope(y, cos, -sin)` vs write a new helper
  -
    - q: Can `wo` (linear) and the MLP actually decode offset from R(ωδ)v_j?
    - a: The output is bilinear in content and offset. A linear readout recovers the offset only along directions where v_j is consistent across tokens. Values have no bias, and latent RMSNorm does not guarantee such a direction, so decoding depends on the downstream SwiGLU. This is unknown empirically; the build stays faithful (content rotation only), and the carrier version is listed below as a separate idea.
    - decides: rotate content only (the idea as written) vs also add a constant carrier pair whose rotated output is a pure offset spectrum
  -
    - q: Is a 45-step kill check long enough to see a recall effect, and is 0.03 nats above seed noise?
    - a: Unknown. 45 steps is about 1.47M tokens. Much of the 2.2-nat recall advantage at step 100 is likely repeated-bigram frequency rather than induction, and whether induction heads have formed by 1.5M tokens at 21M params is unmeasured. The seed-to-seed spread of recall loss is not in the brief. Measure it on 3 baseline seeds before using 0.03 as a kill threshold.
    - decides: kill at 45 steps vs extend the check to 100 steps (or a 10M-token mid-run)
  -
    - q: What do step time and memory cost?
    - a: The extra work is two elementwise rotations per layer on tensors of (4, 6, 1024, ≤32), which is negligible next to the attention and MLP matmuls. Saved activations add roughly 3 MB per layer at bf16, about 40 MB total, well under the 5.6 GB cap. +1% step time is plausible. Use `torch.cat` rather than in-place writes so autograd stays simple. This is a belief from the arithmetic.
    - decides: inline slicing and `torch.cat` vs in-place slice assignment to avoid copies
  -
    - q: Is there prior evidence for rotating values and un-rotating outputs?
    - a: Not from the brief, and none of its sources tests it. My recollection is that Su Jianlin's blog discussed "VO-RoPE" (rotating V and O) in 2025 without a clear LM gain, but I could not verify a URL, so treat this as unverified.
    - decides: build as a leap vs deprioritize
  -
    - q: Does the change survive the harness exam, which builds `GPT(cfg)` from the saved config?
    - a: No parameters or buffers are added, so `model.pt`'s `state_dict` is unchanged. A new `Config` field is only safe if the exam rebuilds `Config` from this file. Hardcode the constants in `Attention`, as `rd = 32` already is.
    - decides: new `Config` fields (`v_rope_dims`, band) vs hardcoded constants in `Attention`
  -
    - q: Should value phases be per-head, given the rotary key is shared across heads?
    - a: The script uses one cache for all heads, so all heads share the value frequencies, and heads differ only through their attention patterns. Per-head frequencies are a separate idea (STRENGTHEN below), not part of the faithful build. This is from the script.
    - decides: one shared cos/sin for all heads vs per-head frequency offsets

known_tagged:
  -
    - text: Head dims: content q/k 64, rotary 32 (shared key `kr` broadcast to 6 heads), v 64; SDPA scale is the default 1/sqrt(96) because q/k are 96-wide
    - prov: script
  -
    - text: `v` is the second half of each head's 128-wide slice of `expand(latent_norm(compress(x)))`; the rotated slice is `v[..., :32]` (or a 16-dim band) taken before SDPA
    - prov: script
  -
    - text: `cos`/`sin` come from `rope_cache(T, 32, theta)` with shape (T, 16) and are built once in `GPT.forward`; reuse them, with no new buffers
    - prov: script
  -
    - text: The inverse rotation is `apply_rope(y_slice, cos, -sin)` applied to `y` in (B, H, T, 64) layout before `transpose(1, 2)` and `wo`
    - prov: script
  -
    - text: Wavelengths of the 16 frequencies run from 2π to 2π·10000^(15/16) ≈ 35,300 tokens; k = 3..10 spans about 35–1990 tokens
    - prov: script
  -
    - text: Attention entropy by layer runs from 0.957 (layer 0) to 0.667 (layer 9); diffuse layers will cancel the rotated content
    - prov: measured
  -
    - text: Recall tokens are 13% of targets, with loss 4.475 against 6.704 for the rest at step 100
    - prov: measured
  -
    - text: Layer-0 attention gradient is anomalously large (0.29 against a neighbour-predicted ~0.04, 3 seeds); watch it after the change
    - prov: measured
  -
    - text: No new parameters and no init changes; Muon/AdamW grouping is unaffected
    - prov: script
  -
    - text: The sliding-window branch is unaffected; the rotation sits outside SDPA, so both branches need it
    - prov: script
  -
    - text: Seed noise on recall-token loss is unmeasured; get it before trusting a 0.03 threshold
    - prov: belief

open:
  -
    - q: What is the seed-to-seed std of recall-token loss at steps 45 and 100?
    - build: False
    - decides: kill threshold 0.03 vs a noise-derived threshold
  -
    - q: Have induction heads formed by 1.5M or 3.3M tokens at this scale?
    - build: True
    - decides: kill check at 45 steps vs 100 steps or mid-run
  -
    - q: Does `wo` plus SwiGLU decode phase from rotated content, or does it need a carrier?
    - build: True
    - decides: content rotation only vs adding a carrier pair
  -
    - q: Is the full 32-dim/16-frequency version better or worse than the k = 3..10 band?
    - build: True
    - decides: 32 rotated dims vs 16
  -
    - q: Does the exam rebuild `Config` from this file or from its own copy?
    - build: True
    - decides: new `config` fields vs hardcoded constants
  -
    - q: Is there a published LM ablation of rotating V and O (VO-RoPE), and what did it find?
    - build: False
    - decides: run as a leap vs deprioritize
  -
    - q: Does damping layer 0's value content reduce or worsen its gradient anomaly?
    - build: True
    - decides: include layer 0 vs exclude it

### Deeper level: Have induction heads formed by 1.5M or 3.3M tokens at this scale?

qa:
  -
    - q: Which training steps do 1.5M and 3.3M tokens correspond to, and where do they sit in the LR schedule?
    - a: With `batch_tokens` 32768 there are 1525 total steps. Step 45 is 1.47M tokens and step 100 is 3.28M. Warmup ends at step 30 (`int(1525·0.02)`), so by step 45 the model has had only about 15 steps at full learning rate. Both checkpoints sit at 3–7% of the run.
    - decides: kill check at step 45 vs a later kill check (≥ step 300)
  -
    - q: Does the measured recall-token gap at step 100 (4.475 vs 6.704) show that induction heads exist?
    - a: No. The split is confounded. Recall targets lean toward frequent tokens and repeated phrases. They can also be lowered by a skip-trigram head that boosts "any earlier token in the window", which is not induction. Without a frequency-matched control, the 2.2-nat gap does not show prefix-matching.
    - decides: recall metric = "bigram already seen in window" split vs a repeated-random-token in-context score
  -
    - q: What does published work say about when induction heads form?
    - a: Olsson et al. find a sharp phase change in small models early in training, at roughly 2.5–5B tokens (https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). That is about 1000× more than 3.3M tokens. Their setup used Adam and a different schedule. Muon-trained speedrun models learn much faster per token, but I know of no published measurement putting natural-text induction formation below about 10M tokens. For this exact configuration the answer is unknown.
    - decides: kill check at step 45/100 vs a check placed after a measured formation step
  -
    - q: Is the model's loss at step 45/100 in the regime where induction usually appears?
    - a: The mixed loss at step 100 is 0.13·4.475 + 0.87·6.704 ≈ 6.41 nats. That is near unigram-level prediction; GPT-2 BPE unigram entropy on web text is roughly 7.3–7.5 nats (my estimate, unchecked). Induction emerges after bigram statistics are learned, at far lower loss. At step 45 the loss is higher still, so the model is still in the unigram/bigram phase.
    - decides: kill check at step 45/100 vs at a loss threshold (e.g. first step with total loss < 4.5)
  -
    - q: Does the precondition exist: a sharp previous-token head in an early layer?
    - a: The measured layer-0 entropy is 0.957 and the minimum across layers is 0.667. Neither is a sharp δ=1 head. Telling δ=1 from δ=2 in the shared 32-dim rotary key needs the fast frequencies. The step at which those entropies were measured is not in the brief, which leaves this unknown. If they were measured at step 100, the previous-token head that K-composition needs has not formed, so induction cannot have formed either.
    - decides: rotate the fast frequencies (k=0..15 over the 32 dims) vs only the slow band (16 dims, k=8..15)
  -
    - q: Before induction forms, does the value rotation cost capacity?
    - a: Yes, this is the expected outcome. With near-uniform attention over T positions, |Σ a_ij e^{iωδ}| ≈ 1/(Tω) for fast ω. The fast-frequency rotated dims therefore cancel to almost nothing, and half of each value head is close to dead early on. At step 45 the idea pays its L2 cost and has nothing to gain from L4. The kill rule's "total loss above baseline" clause is biased toward a false kill.
    - decides: rotate from step 0 vs rotate only after the attention sharpens (ramp or delayed switch)
  -
    - q: Does an induction head need the offset carried in its value at all?
    - a: Not for its own prediction. An induction head's value is the copied token's content, and the offset of the attended token does not change what gets copied. A plausible benefit is upstream instead. A diffuse early head whose output encodes the offset in its phase could hand later layers previous-token information, speeding up induction formation. That benefit would show up as an earlier formation step, not as lower recall loss at a fixed step.
    - decides: rotate values in all layers vs only in early layers (1–3), where the offset can feed previous-token / K-composition
  -
    - q: What cheap measurement settles formation at this scale?
    - a: Train the baseline once and save `state_dict` at steps 45, 100, 200, 400, 800 and 1525. This is a separate probe script, not the harness file. Feed 8×(256 random tokens repeated twice) and measure the loss drop of the second copy against the first. Also compute the per-head prefix-matching score. Formation is the first step where the in-context drop exceeds about 1 nat and some head scores above 0.3. This costs one baseline run of compute.
    - decides: 45-step matched smoke vs a checkpoint-sweep probe before any A/B
  -
    - q: Does this architecture (MLA latent 128, shared rotary key, 4-stream Sinkhorn mixing) delay induction compared with a plain GPT?
    - a: Unknown. Two mechanisms could delay it: - Every head's key content passes through one 128-dim latent shared by 6 heads × (k and v). Previous-token information must survive that compression to be used in K-composition. - The single shared rotary key means all 6 heads see identical positional keys, so per-head positional specialisation comes only from the queries.
    - decides: keep the shared rotary key vs give each head its own rotary key dims for the test
  -
    - q: Which frequency band carries offset information inside a 1024 window without cancelling?
    - a: The wavelength is 2π·10000^(k/16). For k ≥ 12 it is ≥ 6283 tokens, so the rotation over 1024 tokens stays below about 1 rad: little offset information, but little cancellation. For k ≤ 2 (wavelength < 20) it cancels almost completely under diffuse attention. The useful band is roughly k = 3..10 (about 35–1990 tokens). Because `apply_rope` pairs `x[:d]` with `x[d:]`, a 16-dim band must use the matching 8 columns of `cos`/`sin`.
    - decides: rotate `v[..., :32]` with all 16 frequencies vs a 16-dim band using `cos[:, k0:k0+8]`

known_tagged:
  -
    - text: Total steps 1525, accum 8, warmup 30 steps, cooldown starts at step 1067
    - prov: script
  -
    - text: Step 45 = 1,474,560 tokens; step 100 = 3,276,800 tokens
    - prov: script
  -
    - text: Mixed train loss at step 100 ≈ 6.41 (recall 4.475 on 13% of targets, rest 6.704)
    - prov: measured
  -
    - text: ln(50304) = 10.83 nats is the uniform-guess loss; at step 100 the model is about 4.4 nats below it
    - prov: script
  -
    - text: Induction phase change in small models falls at about 2.5–5B tokens under Adam https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
    - prov: web
  -
    - text: Layer-0 attention entropy is 0.957 and the minimum is 0.667 (layer 9); no layer shows a sharp head
    - prov: measured
  -
    - text: `apply_rope` uses half-split pairing; a sub-band rotation must slice `cos`/`sin` to matching columns
    - prov: script
  -
    - text: Rotary frequencies are ω_k = 10000^(−k/16), k = 0..15; k = 3..10 spans wavelengths of about 35–1990 tokens
    - prov: script
  -
    - text: Under uniform attention over T=1024, a rotated pair at frequency ω keeps about 1/(Tω) of its magnitude, so fast bands are near zero early
    - prov: belief
  -
    - text: Induction heads have very probably not formed by step 100 and almost certainly not by step 45 at this scale
    - prov: belief
  -
    - text: The script saves no mid-run checkpoints; a formation sweep needs a separate probe script outside the harness file
    - prov: script
  -
    - text: The layer-0 attention gradient is anomalous (0.29 vs about 0.04); value rotation in layer 0 will change it, so log it
    - prov: measured

open:
  -
    - q: At which step does the baseline form induction heads (repeated-random gain > 1 nat)?
    - build: True
    - decides: kill at step 45 vs at the measured formation step + margin
  -
    - q: At which step were the layer entropies (0.957–0.667) measured?
    - build: True
    - decides: fast band vs mid band only
  -
    - q: How much of the recall-token gap remains under a unigram-frequency-matched control?
    - build: True
    - decides: bigram-seen split vs repeated-random probe as target
  -
    - q: Does the MLA latent (128 shared) delay previous-token head formation compared with standard MHA?
    - build: True
    - decides: shared rotary key vs per-head rotary key for the test
  -
    - q: Can `wo` decode phase-coded offset when it is superposed with content?
    - build: True
    - decides: rotated dims vs explicit offset channel
  -
    - q: Does value rotation in layer 0 change the anomalous layer-0 gradient?
    - build: False
    - decides: watch-only

offshoots:
  - sd643cb4d15
  - seb90bfb8b7
  - s8970d9397d
  - seba836bc5b
  - s8872208160

### New ideas born here
  - sdce62f76f0
  - sce278c5f21
  - sf2547da1ce
  - sbe9e85c46f
  - sd480bde265

### Pre-mortem objections
  -
    - text: Fast rotary pairs scramble the copied content.** The script's `rope_cache(T, 32, 10000)` gives the rotated slice wavelengths of 2π·10000^(k/16): 6.3, 11, 20, 35 and 63 tokens for k=0–4. An induction head copies a token from an offset that varies from window to window. In those pairs the copied value
    - severity: high
    - cls: interaction_cancels
    - link: L4
    - if_true: - metric: learn_z
      - op: above
      - threshold: 0.0
  -
    - text: The phase cannot be decoded without the content.** The output is R(ωδ)v_j. Reading δ from it requires knowing v_j, and v_j varies across tokens. `wo` is linear and the SwiGLU MLP is shallow, so neither can separate content from phase except through products of pairs. The offset signal therefore stay
    - severity: high
    - cls: metric_blind
    - link: L3
    - if_true: - metric: long_z
      - op: above
      - threshold: -1.0
  -
    - text: The cost lands on layer 0, the most important attention layer.** Layers 0–2 have entropy 0.85–0.96. With attention that diffuse, Σa·e^{iωδ} averages the fast pairs toward zero, so half the value dims in those layers become noise. Layer 0's attention gradient is the run's biggest anomaly (0.29, z 7.0
    - severity: high
    - cls: interaction_cancels
    - link: L5
    - if_true: - metric: learn_z
      - op: above
      - threshold: 0.0
  -
    - text: The effect is too small to pass the test.** Recall tokens are 13% of exam targets. Even the chain's own expected gain of −0.02 to −0.10 on recall is only −0.003 to −0.013 nats overall, before any non-recall cost. C7 needs a margin of more than 0.005 at p<0.01 with 3 seeds. With seed standard deviati
    - severity: high
    - cls: below_noise
    - link: L6
    - if_true: - metric: long_z
      - op: above
      - threshold: -2.5
  -
    - text: The early advantage disappears.** At step 100 the baseline has no mature induction circuit, and recall loss is 4.48. Explicit offset phase might give a head start. Over the full run (50M/32768 ≈ 1525 steps), standard previous-token-plus-induction circuits form through the shared rotary key and the 4
    - severity: medium
    - cls: early_only
    - link: L4
    - if_true: - metric: trend
      - op: above
      - threshold: 0.0
  -
    - text: Implementation slip: the inverse rotation is applied after transposing.** `apply_rope` slices `cos[: x.size(2)]`. A builder who inverse-rotates `y` after `y.transpose(1, 2)`, which has shape (B,T,H,D), gets `cos[:6]` of shape (6,16). That broadcasts silently against (B,T,6,16), so the output is rota
    - severity: high
    - cls: impl_slip
    - link: L1
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: Implementation slip: the inverse rotation does not negate sine.** Reusing `apply_rope(y, cos, sin)` for the readout, instead of `apply_rope(y, cos, -sin)`, gives R(ω(i+j)) rather than R(ω(j−i)). That puts absolute position into half of each value head, and the model degrades at positions it rarely s
    - severity: medium
    - cls: impl_slip
    - link: L1
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: The wrong variant was chosen.** The idea text says "first 32 of 64". Only a narrower slice, about 16 dims with wavelengths of 600 to 35k (k≥8), would carry offset at a scale useful within a 1024 context without scrambling content. Building the "first 32" version faithfully includes all 8 fast pairs:
    - severity: medium
    - cls: impl_slip
    - link: L2
    - if_true: - metric: learn_z
      - op: above
      - threshold: 0.0
  -
    - text: The step-time estimate is optimistic.** SDPA cannot fuse the rotation, so each of the 12 layers adds two rotations on (4,6,1024,32) tensors. Each `cat` allocates fresh tensors, and backward saves them. On a 4050 already bound by the 4-stream einsums, the realistic overhead is +3–6%, not +1%. That st
    - severity: low
    - cls: hidden_cost
    - if_true: - metric: step_ratio
      - op: above
      - threshold: 1.05
  -
    - text: bf16 rotation adds noise to half of every value head.** `apply_rope` casts cos/sin to bf16. At positions near 1000, rotating forward then back accumulates relative error of around 2^-8 on every value, including δ=0 self-copies. That adds small noise across all 12 layers, which works against the late
    - severity: low
    - cls: hidden_cost
    - link: L1
  -
    - text: The test metric misses the actual effect.** The KILL criterion uses recall-token loss at 45 steps. If the matched smoke is not paired by seed, recall-loss noise at that point is probably larger than 0.03, so the gate can pass on chance. The exam averages over all tokens, where the non-recall cost do
    - severity: medium
    - cls: metric_blind
    - link: L4
    - if_true: - metric: long_z
      - op: above
      - threshold: 0.0

Pre-mortem verdict: Rotating the fast-wavelength value pairs makes the copied content arrive at an offset-dependent phase that `wo` cannot undo, so induction loses value bandwidth and diffuse early layers (layer 0 especially) lose payload, while the offset signal stays entangled with content and undecodable; the net ex

## Gated nonlinear KV path (latent or expansion)  (part Sequence mixing: latent KV compression and expansion; prior: new)
**Gated latent + gated attention output** (, card deepen, p_success None)
- mechanism: Combine the latent gate with the known per-head sigmoid output gate on the SDPA result (Qiu et al. 2025, gated attention), so selection happens both before keys and values are coded and after retrieval.
- prediction: The gains roughly add up; attention-output norm becomes more uneven across heads.
- test: 2×2 matched smoke (neither, latent gate, output gate, both) at 45 steps.
- source: https://arxiv.org/abs/2505.06708

### Chain and facts

chain:
  -
    - id: L1
    - text: Add `gate = nn.Linear(384,128)` and set latent = `latent_norm(compress(x) * silu(gate(x)))`. This makes the latent channel profile depend on the token rather than being a fixed linear image of it
    - quantity: across-token coefficient of variation of `silu(gate(x))` per channel, plus the share of latent energy in each token's top 16 of 128 channels
    - baseline: unmeasured (the current latent has no gate; the top-16 share of the ungated `latent_norm(compress(x))` is the comparison)
    - expect: CV clearly above 0.1 in most layers; top-16 share up vs the ungated latent at the matched step
    - breaks_if: the gate CV stays below about 0.1, or one channel profile is shared across tokens. Per-token RMSNorm cancels any uniform gate, so such a gate is a no-op.
  -
    - id: L2
    - text: A selective latent gives K/V more usable per-channel information, so attention retrieves context better
    - quantity: recall-token loss (bigram already seen in the window); late-minus-early position gap
    - baseline: 4.475 at step 100; gap +0.459
    - expect: recall loss lower by 0.02 to 0.10 at step 100; gap no wider
    - breaks_if: recall-token loss at step 100 is within ±0.02 of 4.475 over 3 seeds.
  -
    - id: L3
    - text: The extra nonlinearity does not slow early learning. At std 0.02 init, the latent starts roughly bilinear, i.e. not the old linear code
    - quantity: loss at steps 26 and 51
    - baseline: 7.0622, 6.5902
    - expect: within ±0.03 or lower
    - breaks_if: step-26 loss is more than 0.03 above the baseline on all seeds.
  -
    - id: L4
    - text: Better context coding shows up most on rare targets
    - quantity: loss on rank 10000–50304 tokens
    - baseline: 11.613 at step 100
    - expect: 0 to −0.03, smaller in relative terms than the recall-token gain
    - breaks_if: rare-token loss is flat while recall-token loss falls. That would mean the effect is real but mis-targeted.
  -
    - id: L5
    - text: The sum of L2 and L4 gives a lower exam loss after 50M tokens
    - quantity: exam val loss from model.pt
    - baseline: pooled baseline mean (not given here)
    - expect: lower by 0.005 to 0.02 nats
    - breaks_if: one-sided Welch p ≥ 0.01 over 3 or more seeds, or the gain is ≤ 0.005.

weakest:
  L2

qa:
  -
    - q: Is the 128-dim latent a real bottleneck here, or would the same 49k params per layer do more as a wider linear latent?
    - a: Unknown. The only small-scale evidence says a half-rank latent costs about 0.3% loss vs MHA (https://arxiv.org/html/2506.09342v2). Here the latent is a third of d, so the bottleneck is plausible but unproven. The builder should run the equal-param wider latent as the control arm. Without it, a win cannot be credited to nonlinearity.
    - decides: gated 128-dim latent vs ungated latent widened to about 170 (42 × (384+768) = 48.4k params per layer, equal budget)
  -
    - q: silu gate at default std-0.02 init, or 2·sigmoid gate with zero-initialised `W_g` (identical to the current model at step 0)?
    - a: The input x is the RMS-normed read with RMS about 1. `W_g x` therefore has std about 0.02·√384 ≈ 0.39, where silu(z) ≈ z/2 + z²/4. At init the latent is roughly the product of two independent random projections. The linear code the current model starts from is gone, and the latent's sign depends on `W_g x`. RMSNorm cancels the overall 0.5 scale but not this. A zero-init 2·sigmoid gate starts as the current model exactly, because a uniform gate is erased by the latent norm, and then learns relative selection. The spec says silu ("do not water it down"), so build silu. L3 (step-26 loss) is the c
    - decides: `silu(W_g x)` with `_init` std 0.02 vs `2*sigmoid(W_g x)` with `W_g` zeroed
  -
    - q: Which variant: gate on the input before the norm (the idea), or gated expansion `expand(c)*silu(gate(c))` (the alternative)?
    - a: Choose the input gate. Its output passes through `latent_norm`, so V's scale stays at about 0.02·√128 ≈ 0.23 as now. The expansion gate multiplies V, which is not normed, by silu of a std-0.23 signal (≈ 0.12). That shrinks V and the attention output about 8x at init, on top of the already small `wo` init. The expansion gate also costs twice the params (98k per layer).
    - decides: 384→128 gate before `latent_norm` vs 128→768 gate after it
  -
    - q: Can a 45-step smoke run resolve a 0.01-nat difference?
    - a: Probably not. At step 45 (about 1.5M tokens) the loss is around 6.6 and falling fast. Layer-0 attention entropy is 0.957 (near flat) even at step 100, so attention carries little of the loss that early. Seed-to-seed spread at step 45 is unmeasured. The kill should use the step-100 recall-token loss and overall loss across seeds. The step-45 run stays as a crash and VRAM check.
    - decides: kill on mean loss over steps 40–45 vs kill on matched-step loss at step 100–150 over 2–3 seeds
  -
    - q: Should the target metric be rare-token loss or recall-token loss?
    - a: Use recall tokens. The K/V being changed belong to context tokens, and they affect a target only through attention. Rare-target loss (11.613, 12% of tokens) is dominated by unigram and output-embedding learning at this stage. Recall tokens (13% of targets, 4.475) are the attention-dependent class. Keep rare-token loss as a secondary readout.
    - decides: which diagnostic the smoke run logs and kills on
  -
    - q: All 12 layers, or the first few?
    - a: Unknown which wins. The layer-0 input is the RMS-normed token embedding, because the four streams are identical copies and the read weights sum to 1. So the layer-0 gate is a nonlinear per-token K/V code, freed from the tied input/output embedding geometry. Layer-0 attention gradient is also an outlier (0.24–0.29 vs about 0.02 in its neighbours, z 7.03 on 3 seeds). Build all layers as specified, but log per-layer gate CV to see where the gate is used.
    - decides: gate in every block vs gate only in block 0 (or blocks 0–1)
  -
    - q: Should the gate act on the whole shared latent (K and V) or only the V half?
    - a: The idea keeps one latent shared by K and V, so gating it gates both. K then passes through `k_norm`, which partly absorbs the gate. In Qwen's gated-attention study, gating after the value projection helped less than gating the SDPA output. I recall this from https://arxiv.org/abs/2505.06708 and have not re-checked it. Build the shared gate per spec.
    - decides: one shared gated latent vs gating only the V slice of `expand` output
  -
    - q: Which optimizer group does `W_g` land in, and does that matter?
    - a: It is 2-D and its name contains neither "tok" nor "head", so `build_optimizers` puts it in Muon automatically with shape scale max(1, 128/384)^0.5 = 1. The latent RMSNorm makes the forward pass invariant to the gate's overall scale. Its backward pass amplifies gradients by 1/rms(input), but Muon's orthogonalised update hides gradient scale anyway. No change is needed.
    - decides: Muon (automatic) vs explicit AdamW

known_tagged:
  -
    - text: Current attention per layer: `wq` 384×576 (221,184), `compress` 384×128 (49,152), `latent_norm` 128, `expand` 128×768 (98,304), `rotary` 384×32 (12,288), q/k norms 128, `wo` 384×384 (147,456); total 528,640
    - prov: script
  -
    - text: Per block 1,770,160 non-embedding params; 12 blocks plus final norm ≈ 21.242M, which matches the stated 21.24M baseline. The cap is about 23.37M, leaving about 2.12M headroom
    - prov: script
  -
    - text: An MHA baseline with d_ff=1024 also gives 21.24M (1,770,240 per block). Parameter count alone cannot tell whether the measurements are of the MHA baseline or of this MLA script
    - prov: belief
  -
    - text: The gate adds 49,152 params per layer, 589,824 total, giving about 21.83M, inside the cap
    - prov: script
  -
    - text: Plug-in point: `Attention.__init__` adds `self.gate = nn.Linear(cfg.d_model, 128, bias=False)`; `forward` changes `kv = self.expand(self.latent_norm(self.compress(x) * F.silu(self.gate(x))))`. Nothing else changes: `_init` gives std 0.02, and Muon picks it up
    - prov: script
  -
    - text: `latent_norm` is per-token RMSNorm over the 128 channels. Any gate that scales all channels equally for a token has no effect; only relative across-channel gating survives
    - prov: belief
  -
    - text: At init, `gate(x)` has std about 0.39, so silu works in its near-quadratic region, and the latent at init is about half the elementwise product of two random projections
    - prov: belief
  -
    - text: V is not normed: v RMS ≈ 0.02·√128 ≈ 0.23 at init, and the input-gate design leaves this unchanged
    - prov: belief
  -
    - text: Layer-0 attention gradient is 0.2406 vs 0.0195–0.0339 in other layers; the anomaly replicates on 3 seeds (z 7.03)
    - prov: measured
  -
    - text: Recall tokens: 13% of targets, loss 4.475 vs 6.704 for the rest at step 100
    - prov: measured
  -
    - text: Rare tokens (rank 10000–50304): 12% of tokens, loss 11.613 at step 100
    - prov: measured
  -
    - text: First-token attention mass is 0.002–0.006 in all layers, so there is no attention sink to fix
    - prov: measured
  -
    - text: MLA without RoPE trailed MHA by 3–5% at 30M params. The decoupled rotary path here bypasses the latent, so the gate does not touch positional information (https://arxiv.org/html/2506.09342v2)
    - prov: web
  -
    - text: The decoupled shared rotary sub-vector was the only MLA variant that avoided rank collapse in a spectral analysis (https://arxiv.org/html/2507.09394v1)
    - prov: web
  -
    - text: Value-position gating helped less than SDPA-output gating (from memory of https://arxiv.org/abs/2505.06708, not re-checked)
    - prov: belief

open:
  -
    - q: Are the measured baseline numbers from the MHA nano-GPT2 baseline or from this MLA script? Does the MLA script already trail the pooled baseline?
    - build: True
    - decides: gate alone vs gate plus a wider latent to recover an mla deficit first
  -
    - q: Does an equal-param wider linear latent (about 170 dims) match the gated 128-dim latent?
    - build: True
    - decides: gated latent vs widened latent
  -
    - q: What is the seed-to-seed spread of loss at steps 40–45 and at step 100?
    - build: True
    - decides: kill at step 45 vs kill at step 100–150
  -
    - q: Does the silu init (near-bilinear latent) slow the first ~50 steps more than a zero-init 2·sigmoid gate?
    - build: True
    - decides: silu with std-0.02 init vs 2·sigmoid with zero init
  -
    - q: In which layers does the gate learn real across-token variation?
    - build: True
    - decides: all-layer gate vs layer-0 (or early-layer) gate only
  -
    - q: Does gating the K half through the shared latent help or hurt matching, given `k_norm` re-normalises per head?
    - build: True
    - decides: shared k/v gate vs v-only gate
  -
    - q: What is the baseline's peak VRAM at micro_batch 4? It is not reported, though the extra ~38 MB is almost certainly harmless.
    - build: False
    - decides: none unless the baseline is already within ~50 mb of 5.55 gb

dropped_unsourced:
  2

### Deeper level: Are the measured baseline numbers from the MHA nano-GPT2 baseline or from this MLA script? Does the MLA script already trail the pooled baseline?

qa:
  -
    - q: Does the 21.24M baseline parameter count show which script the measurements came from?
    - a: No. The two models are parameter-matched to within about 1k, so the count fits both. MHA at d_ff=1024 has 4×147,456 attention + 3×384×1024 MLP + 768 norms = 1,770,240 per block, 21,243,264 total. This MLA script at d_ff=1077 has 1,770,160 per block, 21,242,304 total. d_ff=1077 is evidently the parameter-matching choice: the 53 extra FFN columns (61,056) offset the MLA saving of 61,184.
    - decides: treat the step-26/51/100 numbers as this MLA script's own baseline vs. re-measure them on this script before any comparison
  -
    - q: Can the loss baselines (7.0622 at step 26, 6.5902 at step 51, recall 4.475 at step 100) be compared with a 45-step smoke run?
    - a: No. `--smoke-steps N` sets `total_steps = N`, and `lr_mult` scales with it. A 45-step run has warmup of 1 step and cooldown from step 31. The full run (1525 steps) has a 30-step warmup and no cooldown before step 1067. A step-100 number can only come from a run of at least 100 steps, so it used a different schedule from the 45-step test. The step-40-to-45 KILL needs a baseline run of this script with `--smoke-steps 45`, on the same seeds.
    - decides: KILL measured against those numbers vs. KILL measured against a fresh baseline run with an identical `--smoke-steps 45`
  -
    - q: Which script produced the layer-0 gradient, attention-mass and recall/rare measurements?
    - a: Unknown; I found no logs. The first-token mass needs explicit attention probabilities, since SDPA does not return them. In this script the probe would have to rebuild `cat(qc_norm, qr)·cat(k_norm, kr)` with the 96-dim scale; an MHA probe would not carry over. The layer-0 gradient spike could also come from the 4-stream embedding broadcast, which the MHA baseline does not have.
    - decides: trust the layer-0 anomaly and the absence of a sink as properties of this MLA + 4-stream model vs. re-probe them
  -
    - q: Does this MLA + mHC script already trail the pooled baseline mean on the exam?
    - a: Unknown. No exam loss for this script is given, and the pooled baseline mean is not given either. Published evidence at small scale is mixed: MLA without RoPE trailed MHA by 3–5% at 30M (https://arxiv.org/html/2506.09342v2). The decoupled-rotary variant used here is the one that avoided rank collapse (https://arxiv.org/html/2507.09394v1), but that paper gives no matched loss figure that I can cite.
    - decides: aim the gate at beating the baseline by 0.005 vs. treat the gate first as recovering an MLA deficit (or drop MLA and gate an MHA key/value path)
  -
    - q: Is the C7 comparator the MHA nano-GPT2 pool or this script?
    - a: The pool is the harness baseline, which the docstring names as nano-GPT2 MHA (nano_specs.md section A). Even so, this script's own 3-seed exam is needed to separate "the gate helps" from "MLA/mHC hurt". If the gate's gain is smaller than an existing MLA deficit, the candidate fails C7 even though the gate works.
    - decides: 3 seeds of this script as a control arm vs. relying only on the harness pool
  -
    - q: Is L1's "gate CV above 0.1" test informative?
    - a: No; it passes trivially at initialisation. The gate input is the normed read, so gate pre-activations have std ≈ 0.02·√384 ≈ 0.39. silu(z) for z ~ N(0, 0.39²) has mean ≈ 0.04 and std ≈ 0.2, which gives a CV of several units before any training. A useful check is the top-16 energy share against the ungated latent and against the gate's own value at step 0, or the gate's correlation with token identity.
    - decides: keep the CV metric vs. replace it with a change-from-init or a per-token selectivity metric
  -
    - q: Does the gate threaten C4 (VRAM or step time)?
    - a: It is unlikely to. At micro-batch 4 the extra activations are about 3 × 4·1024·128 × 2 bytes ≈ 3 MB per layer, about 38 MB over 12 layers. The extra matmul is 384×128 per token, about 2% of block FLOPs. The 3-step smoke is enough, but the step-time ratio must be measured against the baseline, not against this script.
    - decides: run a full VRAM rung vs. a 3-step smoke being enough
  -
    - q: Does the product start near the old linear code, as L3 hopes?
    - a: No. At std 0.02 the latent ≈ RMSNorm(compress(x) ⊙ (gate(x)/2 + gate(x)²/4)), a roughly bilinear code with random signs per channel. RMSNorm restores its scale, so the result is not the old linear image. If L3's early-loss check fails, an offset of about +2 on the gate pre-activation (silu ≈ linear-scaled, near-constant) is the fallback; with `bias=False` that needs an extra parameter of 128 values.
    - decides: plain gate init vs. a gate bias or offset init (silu(g+c)) that starts the latent close to the linear code

known_tagged:
  -
    - text: 21.24M non-embedding matches both MHA at d_ff=1024 (21,243,264) and this MLA at d_ff=1077 (21,242,304), so the count does not show provenance
    - prov: belief
  -
    - text: d_ff=1077 offsets the MLA attention saving of 61,184 with 61,056 extra FFN params
    - prov: script
  -
    - text: Gate adds 49,152 params/layer and 589,824 total, giving about 21.83M against a cap of about 23.37M
    - prov: measured
  -
    - text: Plug-in: `self.gate = nn.Linear(cfg.d_model, 128, bias=False)`; `kv = self.expand(self.latent_norm(self.compress(x) * F.silu(self.gate(x))))`. The gate is 2-D, so Muon picks it up and `_init` gives std 0.02
    - prov: script
  -
    - text: `--smoke-steps N` compresses the LR schedule to N steps: warmup 1, cooldown from step 31 at N=45. Full run: 1525 steps, warmup 30, cooldown from step 1067
    - prov: script
  -
    - text: The baseline step-26/51/100 numbers come from a run of at least 100 steps, with a schedule unknown to me
    - prov: measured
  -
    - text: Gate pre-activation std at init is about 0.39 on the normed read, so the silu output CV across tokens is well above 0.1 before training
    - prov: belief
  -
    - text: The decoupled rotary path (32 dims, shared across heads) bypasses the latent, so the gate does not touch position (https://arxiv.org/html/2507.09394v1)
    - prov: web
  -
    - text: Recall tokens are 13% of targets (4.475 vs 6.704); rare tokens 12% (11.613), all at step 100
    - prov: measured
  -
    - text: The C7 comparator is the harness's pooled baseline mean, which the docstring describes as the MHA nano-GPT2
    - prov: script

open:
  -
    - q: Exam val loss of this MLA script over 3 seeds against the pooled baseline mean
    - build: True
    - decides: gate as a gain on top of the baseline vs. dropping mla and gating an mha key/value path
  -
    - q: Which script and schedule produced 7.0622 / 6.5902 / 4.475 / 11.613
    - build: True
    - decides: reuse those numbers vs. re-measure them on this script
  -
    - q: Seed-to-seed std of the exam loss for this script
    - build: True
    - decides: 3 seeds vs. 5 or more seeds for p < 0.01
  -
    - q: Whether the layer-0 gradient spike comes from the 4-stream broadcast
    - build: True
    - decides: gate all layers vs. layers 1 and up only
  -
    - q: Whether 45-step rankings carry over to 1525 steps
    - build: True
    - decides: 45-step kill vs. 200-step kill
  -
    - q: Gate placement: before the latent norm vs. after the expansion
    - build: True
    - decides: input-side gate vs. gated expansion (the variant)

offshoots:
  - sebdec86283
  - s991a7c939c
  - s368dcc2880
  - se11950c12b
  - s35d1102380
  - s2120347f98

### New ideas born here
  - sb39f34a405
  - sc9319d736c
  - s75b7d6aa2e
  - s1344557338
  - sf8cfc2ca4c

### Pre-mortem objections
  -
    - text: The gate does not touch the real bottleneck. Every head's K and V is still `expand(·)`, a fixed 128-dim column subspace of the 768-dim K/V space. Multiplying the latent by `silu(gate(x))` before `latent_norm` only re-weights coordinates inside that same subspace. Any per-token K/V direction the gate
    - severity: high
    - cls: interaction_cancels
    - link: L2
    - if_true: - metric: long_z
      - op: below
      - threshold: 2.33
  -
    - text: K and V share one gated latent. When the gate saturates a channel off for a token, both its key and its value lose that channel. Conditional selection that helps matching therefore deletes value content, and vice versa. Muon grows the gate weights quickly: an orthogonalized update with spectral norm
    - severity: high
    - cls: interaction_cancels
    - link: L1
    - if_true: - metric: long_z
      - op: below
      - threshold: 0.0
  -
    - text: The realistic effect size is below the exam bar. Gated or nonlinear K/V variants at this scale typically move loss by about 0.002–0.006 nats. C7 requires more than 0.005 nats plus one-sided Welch p<0.01 with 3 seeds. With 3 vs pooled seeds and a seed std of about 0.003–0.006 at 50M tokens, that impl
    - severity: high
    - cls: below_noise
    - link: L5
    - if_true: - metric: long_z
      - op: below
      - threshold: 2.33
  -
    - text: C7 compares against the pooled nano-GPT2 baseline, not against the current MLA + 4-stream script. The current script already swaps full MHA for a 128-dim latent and a single shared 32-dim rotary key broadcast to all heads, and pays for it with d_ff=1077. Its nonembed count (≈21.24M) is matched to th
    - severity: medium
    - cls: interaction_cancels
    - link: L5
    - if_true: - metric: long_z
      - op: below
      - threshold: 0.0
  -
    - text: At std-0.02 init, `gate(x)` has std ≈ 0.02·√384 ≈ 0.39 and silu(z) ≈ z/2. The latent therefore starts as a purely bilinear product of two random projections with no linear term. The baseline's clean linear K/V code, which early previous-token and induction heads need, is gone. The model must first l
    - severity: medium
    - cls: early_only
    - link: L3
    - if_true: - metric: learn_z
      - op: below
      - threshold: -1.0
  -
    - text: Recall-token gains are routed through the wrong path. Copying a bigram seen earlier in the window needs a previous-token head plus an induction head. That is position-driven, and position lives entirely in the shared rotary key `kr = rotary(x)` (32 dims, one head, ungated). The late-minus-early gap 
    - severity: medium
    - cls: interaction_cancels
    - link: L2
    - if_true: - metric: long_z
      - op: below
      - threshold: 1.0
  -
    - text: The rare-token target is metric-blind. Rare-token loss at step 100 (11.613) is above uniform (ln 50304 ≈ 10.83). It is governed by the tied embedding/head pushing down unseen-token logits, which gets about 12% of gradient mass, not by K/V coding. Any change there will come from the head and drift be
    - severity: medium
    - cls: metric_blind
    - link: L4
  -
    - text: Any early edge washes out. Per-token feature selection is exactly what the MLPs in layers 0..i−1 learn to supply to layer i's residual read within a few hundred steps. Whatever advantage the gate gives while the MLPs are immature shrinks over training, and the 30% linear cooldown (which favours the 
    - severity: medium
    - cls: early_only
    - link: L5
    - if_true: - metric: trend
      - op: below
      - threshold: 0.0
  -
    - text: Possible implementation slip: zero-init or near-zero-init of `gate` to "start close to the old model". silu(0)=0, so the latent is ≈0, and `latent_norm` with eps=1e-6 amplifies it by up to 1000×. The result is K/V noise and exploding latent-norm gradients in the first steps, after which `grad_clip=1
    - severity: medium
    - cls: impl_slip
    - if_true: - metric: fidelity
      - op: below
      - threshold: 0.8
  -
    - text: Possible implementation slip: adding the gate behind a new Config flag (for example `latent_gate=False`) that is toggled via the CLI. The harness rebuilds `GPT(cfg)` for the exam. If it uses defaults, or loads with `strict=False`, the gate weights are silently dropped or the load fails, and the exam
    - severity: medium
    - cls: impl_slip
    - if_true: - metric: fidelity
      - op: below
      - threshold: 0.9
  -
    - text: The multiplicative path increases gradient variance where it is already anomalous. Layer-0 attention gradient is 0.29, 7σ above its neighbours on 3 seeds. With `c⊙silu(g)`, the gradient into each factor is scaled by the other factor, so layer 0 receives a noisier, heavier-tailed signal. Muon's norma
    - severity: low
    - cls: interaction_cancels
    - link: L3
    - if_true: - metric: learn_z
      - op: below
      - threshold: 0.0
  -
    - text: Parameter, memory and time costs are small, and they cut against the comparison. The gate adds +0.59M (≈+2.8%, within C3). Compute grows by about 1–3%, and activations by three extra (B,T,128) bf16 tensors per layer, about 40 MB, which is negligible next to the 4× stream residual. Because the gated 
    - severity: low
    - cls: hidden_cost
    - if_true: - metric: step_ratio
      - op: above
      - threshold: 1.05

Pre-mortem verdict: Gating the latent before `latent_norm` leaves every head's K/V confined to the same 128-dim column space of `expand`, and couples K and V to one shared on/off pattern. The upstream MLPs already supply the nonlinear features it adds. The realistic gain is therefore a few thousandths of a nat, below t

## No-affine norms with scalar temperature  (part State across depth: normalization placement and initializati; prior: new)
**No-affine norms with one scalar temperature each (subtract the gain vectors)** (leap, card research, p_success 0.1)
- mechanism: Removal: every per-channel norm gain (9,600 params), which modded-nanogpt shows are unnecessary and which AdamW decay pulls toward zero all run. Reinvestment: each norm keeps a single scalar, initialized so the attention temperature starts at 1.4x and the MLP gate pre-activation at the current scale, because these two temperatures are the only scales Muon's uniform-singular-value updates cannot set cheaply.
- prediction: Loss within 0.01 of baseline through step 100, then lower because the temperatures are no longer fighting decay channel by channel. Params down 9,560.
- test: 45-step smoke; a 500-step run comparing the gain vector's channel spread in the baseline (if the baseline gains stay within 10% of uniform, the vector was idle and the removal is safe).
- breaks: Each normalization owns a learned per-channel gain.
- keeps: Pre-norm placement, RMS statistic, QK-norm, scaled init, all branch shapes.
- kill: step-45 matched loss above baseline by more than 0.03.
- target: update-norm spread, down
- scope: validation loss at 1024
- source: own, builds on https://github.com/kellerjordan/modded-nanogpt no-affine norm

### Chain and facts

chain:
  -
    - id: L1
    - text: Replace per-channel RMSNorm gains with one scalar per norm -> fewer non-embedding params, and per-channel scaling freedom is removed
    - quantity: params_nonembed
    - baseline: 21.24M
    - expect: down 9.5k–12.7k (it depends on which norms are included: 25 scalars give −9,575 and 61 scalars give −12,611; neither matches the stated −9,560)
    - breaks_if: --build-only reports a delta outside that band (the wrong set of norms was converted).
  -
    - id: L2
    - text: A q-side scalar initialized so that q·k starts at 1.4x -> sharper early attention
    - quantity: attention entropy by layer and max attention logit
    - baseline: entropy 0.667–0.957; max logit 7.8–22.75
    - expect: at step 100, mid and late entropy down 0.01–0.08 and layer 0 roughly unchanged; max logit up by less than 3
    - breaks_if: entropy at layers 3–11 changes by less than ±0.01. That would mean the unnormalized rotary part (qr/kr), not the content temperature, sets attention sharpness.
  -
    - id: L3
    - text: The final norm loses its per-channel gain, so the tied head reads exactly the input-embedding geometry. This removes the only per-channel decoupling between the input and output embeddings
    - quantity: matched-step training loss, split by the token-frequency buckets
    - baseline: rank 0–100: 3.163; rank 10000+: 11.613
    - expect: frequent bucket within ±0.02 at step 100; rare bucket within ±0.05
    - breaks_if: the rare-token bucket is worse by more than 0.05 at step 100 on paired seeds.
  -
    - id: L4
    - text: Scalars are exempt from decay (or reach a gradient/decay equilibrium) -> the norm output scale stops drifting downward
    - quantity: trajectory of each scalar, compared with the mean and coefficient of variation (CV) of each baseline gain vector at the end of the run
    - baseline: unmeasured
    - expect: baseline n1/n2 gain CV < 0.15 and the scalars stay within 0.7–1.5
    - breaks_if: the baseline gain vectors end with CV > 0.25 or with channels near 0. That would mean the gains were doing feature selection, not setting a temperature.
  -
    - id: L5
    - text: Fewer AdamW-driven per-channel parameters -> smaller spread in update size
    - baseline: 0.02009 / 0.03265 (x2 spread)
    - expect: max and spread unchanged within ±10%
    - breaks_if: the spread falls by more than 15%. That would be evidence the gains contributed to it. A null result here refutes the stated TARGET, not the idea.
  -
    - id: L6
    - text: All of the above -> exam loss
    - quantity: exam val_loss (3+ seeds) and paired loss_curve
    - baseline: step-100 train loss 6.1789; pooled exam mean unmeasured here
    - expect: mean of steps 36–45 within ±0.03; step 100 within −0.02 to +0.01; final within −0.01 to +0.01
    - breaks_if: the final val is worse by more than 0.005 on the pooled seeds (the idea hurts), or if the 0.005 improvement is not reached at p<0.01 (the idea does nothing, whi

weakest:
  L4

qa:
  -
    - q: Which norms become scalar: only n1/n2/final (which is where the 9,600 count comes from), or also latent_norm, q_norm and k_norm?
    - a: 12×(384+384)+384 = 9,600, so the stated count covers only the stream read-out norms and the final norm. The "attention temperature at 1.4x" can only live in q_norm/k_norm, because n1's output is renormalized before the content logits. So for the mechanism to exist, the q/k norms must be converted. The faithful build is all 37 norms converted, giving 61 scalars. The stated −9,560 matches neither option and should be corrected to −12,611.
    - decides: 25 scalars (−9,575) vs 61 scalars (−12,611)
  -
    - q: What does n1's scalar actually control in this script?
    - a: `qc` is renormalized by q_norm, and k/v pass through compress→latent_norm, which removes n1's scale. n1's scalar therefore reaches only `qr` (via wq) and `kr` (via `rotary`), so the rotary-part logits scale as s². It is a rotary temperature. Keep it, initialized at 1.0, and name it as such.
    - decides: keep a scalar on n1 vs make n1 purely no-affine
  -
    - q: Should the temperature be one learned scalar on q_norm, a scalar on each of q_norm and k_norm, or a fixed `scale=` passed to SDPA?
    - a: The SDPA default scale is 1/√96, since q is 64+32 wide. With unit gains the maximum content logit is therefore 64/√96 ≈ 6.5, and 1.4x makes it about 9.1. Two scalars multiply into one product and add only redundant gradient scaling. One scalar on q_norm, initialized at 1.4, is the clean build. Caveat: the measured max logits already reach 22.75, which is far above 6.5. The rotary part (or gain growth) likely dominates, so the 1.4x may barely matter (see L2).
    - decides: one learned scalar per layer, init 1.4, with k_norm no-affine vs two scalars, each init √1.4
  -
    - q: Should the scalars be weight-decayed?
    - a: `build_optimizers` sends every non-2-D parameter, including 0-D ones, to AdamW with wd 0.1. The decoupled decay factor over 1,525 steps at lr 3e-3, with a mean LR multiplier of about 0.84, is exp(−0.38) ≈ 0.68 in the absence of gradient. The stated mechanism ("no longer fighting decay") only holds if the scalars are exempt, so add a no-decay group.
    - decides: leave them in the existing AdamW group (wd 0.1) vs put them in a new no-decay AdamW group
  -
    - q: Is the final-norm gain redundant?
    - a: The n1, n2 and latent gains can be absorbed into the next linear layer (a diagonal input scaling), so they are redundant in parameterization. The final gain cannot be absorbed, because the head is tied to `tok`. q/k gains cannot be absorbed either, because they come after the normalization. modded-nanogpt's no-gain evidence comes from an untied head, so it does not cover this case. Whether the gain matters here is unknown. Build it faithfully as a scalar and watch the rare-token bucket (L3).
    - decides: scalar final norm (faithful) vs keep the 384-wide final gain as an exception
  -
    - q: How should the scalar be applied?
    - a: The two are mathematically identical. The expand form keeps a single fused kernel; a stride-0 weight may be copied internally, but that is only d floats. The multiply form adds one elementwise pass, about 3 MB of bf16 per call at micro_batch 4, and saves one activation for backward. Use the expand form. The change in VRAM and step time is well inside C4 either way.
    - decides: `F.rms_norm(x, (d,), s.expand(d), eps)` vs `F.rms_norm(x, (d,), None, eps) * s`
  -
    - q: Can a 45-step check with a 0.03 threshold separate signal from noise?
    - a: At equal seed, the data order is identical, because `Shards` is seeded only by `cfg.seed`. The linear-layer init is also identical, provided the scalars are created with `torch.full` or `torch.tensor` (no RNG draws) and module creation order is unchanged. Paired differences are therefore clean. A single-step loss over 32k tokens still carries a few hundredths of noise, so compare the 10-step mean.
    - decides: compare loss at step 45 alone vs compare the mean of loss_curve[35:45] on paired seeds
  -
    - q: Is the proposed TEST ("baseline gains within 10% of uniform → removal safe") informative?
    - a: Uniform shrinkage from decay is not idleness. A small CV does not prove the gains are unused; it only shows a scalar could replace them. The baseline `model.pt` already contains every gain vector at the full budget, so read the per-vector mean and CV there at no cost. The mean of the baseline's q_norm·k_norm product is the empirically learned temperature, and it should set the init instead of 1.4.
    - decides: run a separate 500-step probe vs read the gains from an existing full-budget baseline `model.pt`
  -
    - q: Can removing the gains lower the update-norm spread (the TARGET)?
    - a: Under Adam, the gains move at roughly lr, about 3e-3, relative to a value of about 1, which is below the 0.020 median. They cannot be the 0.0327 maximum. The spread comes from Muon matrices, where the shape scale reaches √6 for `expand` (768×128). Replace the target.
    - decides: keep update spread as the success metric vs replace it with entropy plus paired loss

known_tagged:
  -
    - text: The norms and their widths per block are n1 384, n2 384, latent_norm 128, q_norm 64, k_norm 64, plus a final norm of 384. That totals 12,672 gain parameters.
    - prov: script
  -
    - text: n1's scale reaches only qr/kr (the rotary logits). latent_norm's scale reaches only v, because k is renormalized by k_norm. n2's scale enters both silu(gate) and up, so the MLP output scales roughly as s².
    - prov: script
  -
    - text: SDPA uses its default scale of 1/√96 because q/k are 64 content + 32 rotary dims wide.
    - prov: script
  -
    - text: `read_logits` and `write_logits` are 2-D (2×4) and currently go to Muon; `mix_logits` is 3-D and goes to AdamW. Do not change this in the same build, because it would confound the comparison.
    - prov: script
  -
    - text: 0-D parameters fall into the AdamW "other" group automatically. A new no-decay group requires editing `build_optimizers`.
    - prov: script
  -
    - text: Full run length: 50e6/32768 = 1,525 steps; warmup 30 steps; cooldown starts at step 1,067.
    - prov: script
  -
    - text: The scalars must not draw from torch RNG, or the paired-seed init alignment is lost.
    - prov: script
  -
    - text: Layer-0 attention gradient is 0.24, against about 0.02–0.03 elsewhere (z 7.03, 3 seeds). The likely cause is the tiny raw-embedding stream (std 0.02), and gains do not address it.
    - prov: measured
  -
    - text: Residual RMS grows from 1.403 to 10.893 across depth. Pre-norm read-outs hide this, so the scalars do not see it.
    - prov: measured
  -
    - text: modded-nanogpt applies rms_norm without learned weights. https://github.com/kellerjordan/modded-nanogpt
    - prov: web
  -
    - text: modded-nanogpt's attention scale is about 1.36x of 1/√d_head (0.12 vs 1/√128). This is recalled from the repo and not re-checked.
    - prov: belief
  -
    - text: Gemma's RMSNorm uses a (1+w) gain, so weight decay pulls it toward identity. https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py
    - prov: web

open:
  -
    - q: What per-vector CV and mean do the baseline's n1/n2/final/q/k gains end with at the full budget?
    - build: True
    - decides: scalar init at 1.0/1.4 vs init at the learned means
  -
    - q: Which part of the measured max logits comes from the content half and which from the rotary half?
    - build: True
    - decides: temperature scalar on q_norm vs on n1 (rotary)
  -
    - q: Does the final-norm gain under a tied head carry rare-token information?
    - build: True
    - decides: scalar final norm vs keep the 384-wide final gain
  -
    - q: Does Muon learn diagonal input scalings of wq/w_gate/w_up as fast as an Adam-trained gain does?
    - build: False
    - decides: (interpretation only)
  -
    - q: Is the effect size reachable at all? Is the C7 threshold of more than 0.005 nats plausible for a change of this kind?
    - build: False
    - decides: (whether to spend 3+ seeds vs drop after the smoke test)
  -
    - q: Does a stride-0 expanded weight trigger a copy or a slowdown in F.rms_norm on this torch build?
    - build: True
    - decides: expand-weight form vs post-multiply form

dropped_unsourced:
  1

### Deeper level: What per-vector CV and mean do the baseline's n1/n2/final/q/k gains end with at the full budget?

qa:
  -
    - q: Do the baseline's saved `model.pt` files already hold the answer, so no new training is needed?
    - a: Unknown whether the harness kept them. If it did, `state_dict` holds `blocks.{i}.n1.weight`, `n2.weight`, `attn.latent_norm.weight`, `attn.q_norm.weight`, `attn.k_norm.weight` and `norm.weight` for every seed. Reading them costs nothing and fully settles mean and CV at the full budget. Nothing in the material above contains these numbers.
    - decides: read gains from the existing baseline checkpoints vs spend 3 new full baseline runs on measurement
  -
    - q: How much CV does a gain vector reach from gradient noise alone under this optimizer?
    - a: The gains sit in AdamW with lr 3e-3 and eps 1e-10. With eps that small, Adam is nearly sign-normalized, so even a channel with a tiny, signal-free gradient moves at the full rate. The random-walk displacement is about 3e-3·√(Σ mult²) = 3e-3·√1200.7 ≈ 0.104 at the full budget, or about 0.09 after decay shrinks the early steps. Pure noise therefore gives a CV of about 0.10–0.13. The 10% test threshold sits at or below the noise floor and cannot tell an idle gain from a used one.
    - decides: the idea's "within 10% of uniform means idle" threshold vs a threshold calibrated against the noise floor (about 0.2)
  -
    - q: How far does weight decay alone move the mean?
    - a: The norms are 1-D, so they land in AdamW with weight_decay 0.1. The schedule gives 15.5 + 1037 + 229.5 = 1282 effective full-lr steps. With no gradient, the factor is exp(−3e-4·1282) = 0.681, and it is 0.864 by step 500. Decay is real, but it is weak next to Adam's step size. Equilibrium only needs the gradient signal-to-noise ratio m/√v ≥ 0.1·w, which a scalar (a sum over channels) meets far more easily than a single channel.
    - decides: put the scalars in the decayed AdamW "other" group vs add a no-decay group in `build_optimizers`
  -
    - q: Which gain vectors are functionally redundant, and which are not?
    - a: The n1, n2 and latent_norm gains feed straight into Linear layers (wq/compress/rotary, w_gate/w_up, expand). Their per-channel values are exactly absorbable as column scales of those matrices, so they matter only for optimization: they give Adam a diagonal preconditioner that Muon's orthogonalized updates cannot produce. The q_norm/k_norm gains, and the final gain under tied embeddings, are not absorbable. They set the logit geometry directly.
    - decides: the 25-scalar set (n1, n2, final) vs the 61-scalar set (every RMSNorm instance)
  -
    - q: Where must the "1.4x attention temperature" scalar sit to actually reach q·k?
    - a: The overall scale of n1 cancels in qc (q_norm), k (k_norm) and v (latent_norm). It survives only in qr and kr, both of which come from n1, so an n1 scalar s scales the rotary logits by s² and leaves the content logits untouched. A 1.4x content temperature therefore needs a scalar on q_norm (or k_norm). q_norm is outside the 9,600-parameter set the idea names, so the 25-scalar build cannot deliver L2.
    - decides: scalar on q_norm (content temperature) vs scalar on n1 (rotary temperature only)
  -
    - q: Which statistic of the q/k gains is meaningful?
    - a: The content logit depends only on the per-channel product gq⊙gk. Each 64-vector is shared across all 6 heads of its layer, because RMSNorm(64) acts on the last dimension. The individual vectors can drift along gq·gk = const, with decay pulling them toward gq = gk. The CV of each vector alone is therefore ambiguous: measure the CV and mean of the product. One scalar on q captures the whole temperature; a second scalar on k is a redundant direction.
    - decides: one temperature scalar on q only vs a scalar on both q and k
  -
    - q: Does a 500-step probe predict the full-budget end state?
    - a: No. At step 500 the noise CV is about 0.066 and the decay factor 0.864. At the end they are about 0.10 and 0.681. Any real per-channel drift also keeps growing during the 1,037 flat steps. A 500-step measurement underestimates both mean drift and CV, so the open question requires the full budget, or Q1's checkpoints.
    - decides: a 500-step baseline probe vs a full 1,525-step probe
  -
    - q: How do you tell feature selection from temperature in the end state, beyond CV?
    - a: Count the channels with |g| < 0.2·mean and check for sign flips, since Adam can push a gain negative. Then mean-ablate each vector in the checkpoint (replace it by its mean) and measure the loss change on train-shard windows split by token-frequency bucket. Never use val. For the absorbable gains (n1/n2/latent), ablation overstates their importance because the downstream matrices co-adapted. For the final and q/k gains, the ablation delta is the direct answer.
    - decides: convert the final norm to a scalar vs keep the final gain vector
  -
    - q: What should the scalars be initialized to?
    - a: Unknown until Q1 or the probe is answered. Initializing at 1.0 keeps step-0 behavior identical to the baseline, so it preserves the paired comparison. Initializing at the end-state mean (if it differs from 1 by more than 0.2) removes a drift the scalar would otherwise spend early steps on. Either way, `torch.tensor(value)` draws no RNG.
    - decides: init 1.0 (the current start) vs init to the baseline's end-of-run mean
  -
    - q: Muon decays matrices hard. Do the gains carry the scales Muon cannot hold?
    - a: Muon's decay alone would give exp(−0.02·0.1·1282) = exp(−2.56) ≈ 0.077, so Muon matrix scale sits at an lr/wd equilibrium that ignores the gradient. That leaves the n2 gain as the free knob for the gate pre-activation scale, and the q/k gains as the only knob for content temperature. This supports the idea's premise that the temperatures matter. It also predicts that baseline n2 and q/k means drift away from 1, a belief not yet measured.
    - decides: a plain scalar vs a Gemma-style (1+w) vector that decays toward identity

known_tagged:
  -
    - text: The norm gains (n1, n2, latent_norm, q_norm, k_norm, final) are 1-D, so they go to AdamW with lr 3e-3, betas (0.9, 0.95), eps 1e-10 and weight_decay 0.1. No norm is exempt from decay.
    - prov: script
  -
    - text: Effective full-lr steps over the run: warmup 15.5, flat 1,037, cooldown 229.5, total 1,282. Σ mult² ≈ 1,200.7.
    - prov: script
  -
    - text: The `RMSNorm` class is shared by all 61 instances. Editing the class converts all of them (−12,611 params). The 25-scalar set (−9,575) needs a separate class or a flag. The stated −9,560 matches neither.
    - prov: script
  -
    - text: The q_norm and k_norm gains (64 each) are shared across the 6 heads and touch only the 64 content dims. The 32 rotary dims are never normalized.
    - prov: script
  -
    - text: An n1 scalar s scales only the rotary logits, by s². The content temperature lives only in gq⊙gk.
    - prov: script
  -
    - text: The n1, n2 and latent gains are exactly absorbable into the columns of the following Linear layers. The q/k and final gains are not, the final one because of the tied head.
    - prov: script
  -
    - text: Muon's decay factor under its own lr and wd would be about 0.077 over the run, so matrix scales are set by equilibrium. [belief, derived from the script]
    - prov: untagged
  -
    - text: The baseline end-state gain mean and CV are not measured anywhere in the material provided. [measured: absent]
    - prov: untagged
  -
    - text: Scalar parameters made with `torch.tensor(...)` draw no RNG, so paired-seed init alignment holds.
    - prov: script
  -
    - text: The `state_dict` keys change, but the exam builds `GPT(cfg)` from this script, so loading stays consistent.
    - prov: belief
  -
    - text: Gemma's (1+w) gain makes decay pull toward identity: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py
    - prov: web
  -
    - text: modded-nanogpt uses weightless rms_norm: https://github.com/kellerjordan/modded-nanogpt
    - prov: web

open:
  -
    - q: What are the end-of-run mean and CV per vector of n1, n2, latent, gq⊙gk, and final across the 3 baseline seeds?
    - build: True
    - decides: 25-scalar vs 61-scalar set; init 1.0 vs init at the measured mean
  -
    - q: Did the harness keep the baseline `model.pt` files?
    - build: True
    - decides: read checkpoints vs rerun the baseline
  -
    - q: Does mean-ablating the final gain hurt the rare-token bucket (rank 10000+) by more than 0.05 on train windows?
    - build: True
    - decides: scalarize the final norm vs keep its vector
  -
    - q: Is the 1.4x scalar meant for q_norm (content) or n1 (rotary)? The idea text is ambiguous.
    - build: True
    - decides: q_norm scalar vs n1 scalar
  -
    - q: Are there sign-flipped or near-zero channels (
    - build: True
    - decides: scalar (temperature) vs keep the vector (feature selection)
  -
    - q: What does L5's quantity mean? The field is garbled: probably the median and max of AdamW per-parameter update norms.
    - build: False
    - decides: which update-norm statistic to log vs dropping l5

offshoots:
  - sfa8fafb470
  - sf2f3f8a5c7
  - s41dfb288b7
  - s19aa8ba652
  - s506ad35bf5

### New ideas born here
  - sa8fe9720aa
  - sf5b6a37334
  - s1964c2ddb5
  - s86c9b40fa3
  - se66618a9fe
  - s3823e432c5

### Pre-mortem objections
  -
    - text: The idea's own L6 prediction (final within −0.01 to +0.01) cannot pass C7. Clearing "beats the pooled mean by >0.005 at p<0.01, one-sided Welch, 3 seeds" needs a true effect of roughly 2–3x the per-seed std of exam loss. Removing 12.6k of 21.24M non-embedding params (0.06%) sits deep inside seed noi
    - severity: high
    - cls: below_noise
    - link: L6
    - if_true: - metric: long_z
      - op: below
      - threshold: 2.33
  -
    - text: "Temperatures no longer fighting decay channel by channel" has no mechanism here. A scalar in AdamW (wd 0.1, lr 3e-3, so 3e-4/step shrink) feels exactly the same decoupled decay as each gain channel. Adam's per-element normalization also gives a scalar the same ~lr-sized step as one channel did, so 
    - severity: high
    - cls: interaction_cancels
    - link: L4
    - if_true: - metric: long_z
      - op: below
      - threshold: 1.0
  -
    - text: The 1.4x attention temperature only helps early. The baseline's q_norm/k_norm gains are AdamW-trained at lr 3e-3. They reach a product of ~1.4 within ~100–150 of the 1525 steps, and the cooldown (last 30%) dominates final loss. Any head start is gone by mid-run. Check: the paired loss-difference cur
    - severity: high
    - cls: early_only
    - link: L2
    - if_true: - metric: trend
      - op: below
      - threshold: 0.0
  -
    - text: The content temperature is not what sets attention sharpness. With RMS-normalized qc and k (unit gains, 64 dims) and SDPA scaling by 1/sqrt(96) (q's last dim is hd+rd=96), the max content logit is about 64/9.8≈6.5. Even with gains grown ~1.25 each by step 100 it is about 10. The measured max logits 
    - severity: medium
    - cls: interaction_cancels
    - link: L2
    - if_true: - metric: learn_z
      - op: below
      - threshold: 1.0
  -
    - text: Most converted norms are redundant with downstream Muon matrices or sit before scale-invariant ops. latent_norm's gain is followed by `expand`, whose output k goes through k_norm again. n1's output passes through wq→q_norm and compress→latent_norm, which discard its scale; only qr/kr see it. n2's sc
    - severity: medium
    - cls: interaction_cancels
    - link: L4
    - if_true: - metric: long_z
      - op: below
      - threshold: 1.0
  -
    - text: Removing the final-norm gain vector under tied embeddings removes the only per-dimension decoupling between input lookup and output scoring. Modded-nanogpt drops gains with an untied lm_head, so its evidence does not transfer. The cost accumulates in the rare buckets (rank 1000+, 38% of tokens, loss
    - severity: medium
    - cls: early_only
    - link: L3
    - if_true: - metric: long_z
      - op: below
      - threshold: 0.0
  -
    - text: Implementation slip that silently changes the optimizer. `build_optimizers` routes every `ndim==2` parameter not named tok/head to Muon; this already catches read_logits and write_logits (shape 2x4). If the builder stores the scalar as shape (1,1) for broadcasting, it lands in Muon. Newton–Schulz on
    - severity: medium
    - cls: impl_slip
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: Implementation slip that doubles the temperature. Initializing "q·k at 1.4x" by setting 1.4 on both q_norm and k_norm gives 1.96x. Placing it on n1 instead does nothing to the content logits, because q_norm renormalizes, but scales qr·kr by 1.96x. Either way, early attention is much sharper than int
    - severity: medium
    - cls: impl_slip
    - link: L2
    - if_true: - metric: learn_z
      - op: below
      - threshold: -1.0
  -
    - text: The parameter delta doesn't match any consistent set of norms. Per block, n1+n2+latent+q+k = 1024 gain params; with the final norm the total is 12,672. Converting all 61 gives −12,611, and n1/n2/final only (25) gives −9,575. Neither equals the stated −9,560, which implies 40 scalars, a set no partit
    - severity: low
    - cls: impl_slip
    - link: L1
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: The stated TARGET (update-norm spread) is blind to this change.
    - severity: low
    - cls: metric_blind
    - link: L5
  -
    - text: Grad-clip interaction. A scalar's gradient is the sum of its channels' gradients, so for a coherent temperature signal its norm is up to ~sqrt(384)≈20x the gain vector's. That raises the global norm and triggers clip=1.0 more often. Muon (NS-normalized) and Adam (eps 1e-10) are almost scale-invarian
    - severity: low
    - cls: hidden_cost
    - if_true: - metric: learn_z
      - op: below
      - threshold: 0.0
  -
    - text: Hidden costs are trivial. The extra multiply per norm (or `F.rms_norm(x, ..., None) * s`) adds no measurable step time or VRAM, so the change also cannot be "paid for" by reinvesting a saving elsewhere.
    - severity: low
    - cls: hidden_cost
    - if_true: - metric: step_ratio
      - op: above
      - threshold: 1.02

Pre-mortem verdict: The scalars sit in the same AdamW decay regime the gains did, the 1.4x temperature head start is erased within ~150 of 1525 steps, and the rotary path, not the normalized content path, sets attention sharpness. The net effect at the exam is within ±0.005, far below what 3 seeds at p<0.01 can resolve

## Per-token scale and bias decoupling norm from direction  (part Inputs and outputs: tied embedding, stream read-out, logits; prior: new)
**Direction-only tie with per-token scale and bias** (leap, card research, p_success 0.15)
- mechanism: Keep the shared 50304x384 table but use its RMS-normalized rows as the output classifier, multiplied by a learned per-token log-scale and plus a learned per-token bias (two vectors of 50304 scalars, zero-init). The tie currently forces one row norm to serve as input salience and as output frequency prior at once, and the 2026 tying analysis shows the output gradient wins that fight; decoupling norm from direction lets the shared direction serve both roles while rare rows (rank 10k+, 11.6 nats) keep a usable direction even when decay shrinks their norm. The conflict is intrinsic to tying and do
- prediction: Loss curve pulls below baseline within the first few hundred steps as the bias absorbs the unigram prior, and the gap persists; parameters rise by about 100k to roughly 21.34M; step time within 2 percent (one 50304x384 normalize per forward, done in bf16 to stay under the 0.1 GB memory headroom); th
- test: Build-only for the count, 3 steps for time and peak memory, then 45 matched steps against baseline and against the bias-only variant below to isolate the normalization.
- breaks: The tied row's norm must encode both input salience and output frequency; this flips it so norm is output-only and lives in two cheap scalar vectors.
- keeps: One shared table, plain next-token cross entropy, final RMSNorm, mean read-out, the same Muon and AdamW grouping (the two new vectors fall to AdamW as 1-D parameters).
- kill: Mean training loss over matched steps 30 to 45 not at least 0.02 nats below baseline, or below the bias-only variant by less than 0.005.
- target: loss on rare tokens, down
- scope: validation loss at 1024
- source: own, building on https://arxiv.org/pdf/2603.26663

### Chain and facts

chain:
  -
    - id: L1
    - text: Head gets a per-token bias b initialised to counted log-unigram, kept out of weight decay -> the frequency prior is in the logits from step 0 instead of being learned through row geometry
    - quantity: training loss at matched steps 1–26
    - baseline: 10.88 at step 1, 7.06 at step 26
    - expect: down 1–3 nats at steps 1–10, down 0.03–0.3 at step 26; a zero-init learned bias gives under 0.05 at step 26
    - breaks_if: counted-init gap at step 26 is under 0.02 nats
  -
    - id: L2
    - text: Bias carries the prior -> rows stop sharing a common "push-down" direction
    - quantity: cosine of each output row to the mean output row (anisotropy), and ‖mean row‖ / mean ‖row‖
    - baseline: unmeasured
    - expect: down by 20–60% relative at step 100
    - breaks_if: anisotropy at step 100 is within 10% of baseline
  -
    - id: L3
    - text: Rows normalised in the head -> the output gradient has no radial component, so row norm is set only by the input lookup and decay
    - quantity: Spearman correlation of row norm with log train frequency
    - baseline: unmeasured (belief: strongly positive)
    - expect: falls toward 0 to 0.3
    - breaks_if: correlation stays above 0.5 at step 100
  -
    - id: L4
    - text: Input rows no longer carry the output prior -> early layers get cleaner token features, so mid-frequency conditional prediction improves beyond what bias alone gives
    - quantity: loss on rank 100–1000 and 1000–10000 bands, compared with the bias-only variant
    - baseline: 6.829 / 8.979 at step 100
    - expect: down 0.01–0.1 compared with bias-only
    - breaks_if: direction-only minus bias-only is above −0.005 on both bands over 3 seeds
  -
    - id: L5
    - text: Early gain persists through 1525 steps -> exam loss falls
    - quantity: training loss at steps 1000–1525 and exam val loss
    - baseline: unmeasured here
    - expect: down 0.005–0.03 nats
    - breaks_if: the matched training-loss gap over the last 300 steps is under 0.005

weakest:
  L4

qa:
  -
    - q: With zero-init log-scale s and RMS-normalised rows, what is the initial logit scale?
    - a: RMS-normalised rows have RMS 1, which is 50x the baseline row RMS of 0.02. Logit std becomes about √384 ≈ 19.6 instead of 0.39, and the output gradient into the tied table is also amplified 50x. Zero-init therefore starts at a loss of tens of nats and makes the output side dominate even more, the opposite of the intent. Initialise s = log(0.02) so the head equals the baseline head at step 0, or equivalently L2-normalise and set s = log(0.392).
    - decides: s init = log(0.02) ≈ −3.91 vs s = 0
  -
    - q: Can a zero-init learned bias absorb the unigram prior "within the first few hundred steps" under AdamW lr 3e-3?
    - a: No. Adam moves each element by roughly the learning rate or less per step. That is at most about 0.9 nats by step 300 and about 3.5–4.5 nats over the whole schedule. The log-prior spans more than 10 nats between frequent and rank-10k+ tokens. The prediction in the idea cannot happen with zero init, so take the counted-init variant.
    - decides: zero-init learned b vs b initialised to counted log-unigram
  -
    - q: Should s and b be in the AdamW group with weight_decay = 0.1?
    - a: Decoupled decay costs lr·wd = 3e-4 per step times the value. For b ≈ −10 that is 3e-3 per step toward 0, the same size as the Adam step, and it erases roughly a third of the prior over the run. Decay on s pulls it toward exp(0) = 1, which is a 50x temperature. Put both in a no-decay AdamW group.
    - decides: the existing `other` group vs a new no-decay group
  -
    - q: How should the per-token scale and bias be applied without breaking C4?
    - a: Logits are 4096×50304 in bf16, about 412 MB. Any elementwise operation on them saves at least one more such tensor for backward, which is far beyond the roughly 0.1 GB headroom. Folding adds one 50304×384 bf16 tensor (38.6 MB) plus row statistics. The bias rides in the addmm, so no extra logits-sized tensor is created.
    - decides: fold into the weight (`W = E * (exp(s) / rms_row(E))[:, None]`, then `F.linear(h, W, b)`) vs elementwise scale/add on the logits
  -
    - q: Does normalisation beyond the bias change anything the optimiser sees?
    - a: The output gradient to a raw row becomes (exp(s)/rms)·(I − êêᵀ)g. The radial part is removed exactly, and the tangential part shrinks as rows grow above their initial norm. Under per-element Adam normalisation, magnitudes are partly washed out, so the main surviving effect is the loss of the radial push. Whether that lowers loss is unknown, and it is what the bias-only arm tests.
    - decides: RMS-normalised direction-only head plus s and b vs bias-only
  -
    - q: Is "rare-token loss down" the right target?
    - a: At step 100 the rank-10k+ band sits at 11.6 nats, already above uniform (10.83), so the model is already charging roughly a prior cost there. A correct unigram prior gives rank-10k+ tokens their unigram code length, which is plausibly 12–15 nats (belief). That can raise rare-band loss while lowering total loss. Judge on total loss and the mid bands, and only report the rare band.
    - decides: rank-10k+ band loss vs total loss plus the 100–10000 bands as the success signal
  -
    - q: Where are the counts computed?
    - a: The exam runs GPT(cfg) and then load_state_dict, so counting inside __init__ would read data at exam time and break --build-only. Count in main after the build-only return: - Source: `train.data[0]` with `np.bincount(..., minlength=50304)`. - Smoothing: add-one. The 47 padding ids then get finite, very negative values; avoid −inf in bf16. - Store with `b.copy_(log p)`. Because b is a Parameter, the exam sees it. Never touch val.bin.
    - decides: inside GPT.__init__ vs in main after the --build-only return
  -
    - q: Is the steps 30–45 kill window meaningful?
    - a: Baseline is already at 7.06 by step 26, likely at or below unigram entropy (belief: about 7.3–7.6 nats for GPT-2 BPE on FineWeb-Edu). So steps 30–45 already test whether the head start survives into the bigram phase. However, the 0.02 bar is 4x the C7 margin, and early prior gains routinely fade. Keep 30–45 as the first gate and add a 300-step matched check before any full run. Matched seeds give identical data order, because the Shards RNG is independent of the model.
    - decides: kill on steps 30–45 vs kill on a matched window at about steps 250–300
  -
    - q: Does the bias interact with global grad clipping at 1.0?
    - a: Unknown magnitude. The bias gradient is mean(p − onehot) over 32k tokens. Its norm is up to about 1 when the prior is wrong and small when it is counted. With counted init it should rarely trigger clipping. Log the pre-clip global norm in both arms.
    - decides: include s and b in clip_grad_norm_ vs clip them separately or exclude them
  -
    - q: Does the tied head's self-logit (the token scoring itself through its own embedding in the residual) get worse?
    - a: Sinkhorn mixing is doubly stochastic, so the mean over streams preserves the input exactly: the readout always contains e_x plus the branch writes. With s = log(rms0), the self-logit at init equals baseline (about 7.7). Later, as raw rows grow on the input side but not in the head, the ratio drifts in an unmeasured direction. Measure the logit of the current token in both arms.
    - decides: plain readout vs readout that subtracts tok(idx)

known_tagged:
  -
    - text: The head is `nn.Linear` with weight tied to `tok.weight`, applied to `self.norm(x.mean(dim=2))`. Logits stay bf16 under autocast and the loss is plain F.cross_entropy.
    - prov: script
  -
    - text: `param_counts` subtracts only `tok.weight`, so s and b (2 × 50304 = 100,608) count as non-embedding: 21.24M → about 21.34M, cap about 23.36M. A frozen buffer would not count.
    - prov: script
  -
    - text: `named_parameters` dedupes the tie, so only `tok.weight` appears; it goes to AdamW (lr 3e-3, wd 0.1, betas 0.9/0.95). New 1-D vectors fall to AdamW too, but need their own wd = 0 group.
    - prov: script
  -
    - text: Steps = 50e6 / 32768 = 1525; accum = 8 micro-batches of 4×1024; warmup 2%, cooldown 30%.
    - prov: script
  -
    - text: Initial row RMS is 0.02 and RMSNorm output RMS is 1. Initialise s = log(0.02) for RMS-normalised rows so step-0 logits match baseline.
    - prov: script
  -
    - text: Logits tensor per micro-batch is 4096×50304 bf16 ≈ 412 MB. Never add a logits-sized elementwise op; fold s into the weight and b into F.linear.
    - prov: script
  -
    - text: The exam builds GPT(cfg) and loads `state_dict`. The prior must be a Parameter or persistent buffer filled in main before training, never computed in `__init__`.
    - prov: script
  -
    - text: Mean over Sinkhorn-mixed streams preserves the input, so the readout always contains exactly e_x plus branch writes.
    - prov: script
  -
    - text: At step 100, the rank-10k+ band is 11.613 nats on 12% of tokens; the rank 0–100 band is 3.163 on 42%; step-26 loss is 7.06.
    - prov: measured
  -
    - text: Layer-0 gradient is anomalously high (attn 0.24–0.29, about 10x layer 1) on 3 seeds. It is a possible readout of input-side pressure; watch it in both arms.
    - prov: measured
  -
    - text: Tied matrices end up output-shaped: mean cosine 0.72 to the untied output matrix vs 0.53 to the input matrix (OLMo-1B scale) https://arxiv.org/pdf/2603.26663
    - prov: web
  -
    - text: Centering and z-loss leave final loss unchanged (3.05 at 221M) and only reduce LR sensitivity https://arxiv.org/html/2601.02031v1
    - prov: web

open:
  -
    - q: Does the baseline row norm actually correlate with log frequency at step 100?
    - build: True
    - decides: per-token scale s included vs bias-only head
  -
    - q: What is the unigram entropy of FineWeb-Edu under GPT-2 BPE (for example 7.3 vs 7.6 nats)?
    - build: False
    - decides: kill window at steps 30–45 vs a later window
  -
    - q: What is the matched-seed noise of a 15-step mean loss across seeds, and can a 0.005 bias-only margin be resolved with 3 seeds?
    - build: True
    - decides: kill on steps 30–45 vs a 300-step window
  -
    - q: Should the bias be learnable after the counted init, or frozen as a buffer?
    - build: True
    - decides: parameter (counts toward c3) vs persistent buffer
  -
    - q: RMS vs L2 normalisation with matched s init: is there any difference beyond a constant?
    - build: False
    - decides: `f.rms_norm` rows vs `f.normalize` rows
  -
    - q: Does the self-logit of the current token drift after decoupling?
    - build: True
    - decides: plain readout vs readout minus tok(idx)
  -
    - q: Does a one-shard count differ meaningfully from a full-train count for rank-10k+ tokens?
    - build: True
    - decides: count from `data[0]` only vs from all shards (cost: startup seconds)

dropped_unsourced:
  1

### Deeper level: What is the matched-seed noise of a 15-step mean loss across seeds, and can a 0.005 bias-only margin be resolved with 3 seeds?

qa:
  -
    - q: Does a 45-step smoke run reproduce the full run's learning-rate schedule at steps 30–45?
    - a: No. With `--smoke-steps 45`, `lr_mult` gets warm = max(1, int(0.9)) = 1 and cool_start = int(31.5) = 31. So indices 31–44 decay linearly from 1.0 to 1/14, and the KILL window would be mostly annealed loss. In the full run, warm = 30 and cool_start = 1067, so steps 30–45 are the first full-LR steps. `--smoke-seconds` keeps total_steps = 1525 and only breaks on time, so its `loss_curve[:45]` matches the full run step for step. Use it, with a time budget that covers at least 45 steps, and read `loss_curve[29:45]`.
    - decides: `--smoke-steps 45` vs `--smoke-seconds` cutoff with the 1525-step schedule left in place
  -
    - q: Do arms with the same seed see identical batches?
    - a: Yes, provided the build keeps them identical. `Shards` draws from its own `np.random.default_rng(seed)`, which only `sample()` consumes, so batch k is the same in every arm with the same seed. Torch init is also identical as long as s and b are created with `torch.zeros`/`torch.full` (no RNG calls). The counted prior must be computed from a fixed slice, for example the first N tokens of `train_000`, never through `train.rng`. Under these conditions, batch-content noise cancels in the per-seed arm difference.
    - decides: paired same-seed differencing vs pooled unpaired Welch for the KILL rule
  -
    - q: Is a same-seed rerun bitwise reproducible, so that the paired noise floor is zero?
    - a: Unknown. The tied table receives an embedding-backward gradient and an SDPA backward gradient, and PyTorch lists several CUDA backward ops as nondeterministic (https://pytorch.org/docs/stable/notes/randomness.html). Under bf16 and Muon at full LR, any bit difference will grow. Two same-seed baseline reruns give the paired floor directly: the sd of their `mean(loss_curve[29:45])` difference.
    - decides: measure the floor with A/A same-seed reruns vs force `torch.use_deterministic_algorithms(True)` for the probe runs
  -
    - q: What is the across-seed sd of the 15-step mean training loss?
    - a: Unknown here. It can be computed at no cost from the three existing baseline seeds' `loss_curve` (the layer-0 gradient measurement already used 3 seeds). My belief: 32 windows per step with a window-level loss sd of about 0.4–0.5 nats gives about 0.07–0.09 per step. Seed-dependent data then makes the 15-step mean sd about 0.015–0.025 nats, plus init variance.
    - decides: an unpaired 3+3 Welch kill vs a paired or longer-window kill
  -
    - q: What sd would let 3 seeds resolve a 0.005 margin?
    - a: Unpaired Welch with 3+3 and equal variance has df ≈ 4 and t₀.₉₉ = 3.747. SE = 0.816·s, so it needs s < 0.0016 nats, which is about 10× below my belief in A4. That is not resolvable. Paired with n = 3 (df 2, t₀.₉₉ = 6.965) needs sd_d < 0.00124. With n = 5 (t = 3.747) it needs sd_d < 0.0030; with n = 8 (t = 2.998), sd_d < 0.0047. Three seeds resolve 0.005 only if the A/A floor from A3 is about 0.001 or less.
    - decides: 3 seeds vs 5–8 seeds (or a paired design)
  -
    - q: Is the logged loss precise enough to see 0.005?
    - a: Probably yes. `log_softmax`/`nll_loss` are on autocast's float32 list (https://pytorch.org/docs/stable/amp.html), so the loss should be fp32, and `loss_curve` rounds to 1e-4. If it were bf16, the grid spacing in the 4–8 range would be 0.031. Print `loss.dtype` once to confirm.
    - decides: use `loss.item()` as is vs log an explicit fp32 loss
  -
    - q: Is steps 30–45 the right phase for the bias-only vs direction-only margin?
    - a: Probably not for L4. The baseline is already at 7.06 by step 26, which I believe is below the GPT-2-token unigram entropy of about 7.5 nats (unchecked). The unigram prior is therefore largely learned by step 30, and any bias advantage there is a transient. The representation effect that L4 claims would show later, and the A/A noise floor also grows with step count. Both windows need the A3 measurement.
    - decides: KILL window 30–45 vs a later window (e.g. 200–300 at full LR)
  -
    - q: Does the bias gradient change grad clipping, confounding the comparison against the baseline?
    - a: Possibly. `clip_grad_norm_` covers every parameter, and b's gradient (mean softmax minus empirical frequency) has norm up to √2. Whenever the baseline's total norm is near 1, adding b scales down every other update. This is shared by bias-only and direction-only, but it confounds the comparison with the baseline. Log the value `clip_grad_norm_` returns in all arms.
    - decides: clip over all params vs exclude s and b from `clip_grad_norm_`
  -
    - q: Should the readout be the training-loss window or a fixed probe set?
    - a: A fixed probe gives a paired, step-free readout plus per-band rare-token losses for L4 at no extra cost. Example: 64 windows from the tail of the last `train_*.bin`; the sampler's overlap with them is negligible, and val is never touched (C6). However, smoke runs save no `model.pt`, so the probe has to live in the script (train-only) or a separate tool needs a weights dump. Unknown which the harness tolerates.
    - decides: `loss_curve[29:45]` mean vs a fixed probe of train windows evaluated after step 45
  -
    - q: Must the bias-only arm share the direction-only arm's bias init?
    - a: Yes. L1 predicts that counted vs zero init alone moves step-26 loss by 0.03–0.3. That is 6–60× the 0.005 margin, so mismatched inits make the KILL test measure init instead of normalization. Use counted init in both arms, or zero in both.
    - decides: zero-init bias-only vs counted-init bias-only as the comparison arm

known_tagged:
  -
    - text: `--smoke-steps 45` gives warm = 1 and cool_start = 31; the full run gives warm = 30 and cool_start = 1067 out of 1525 steps. Steps 30–45 have different LR in the two modes.
    - prov: script
  -
    - text: `--smoke-seconds` keeps the 1525-step schedule, so `loss_curve[:45]` is schedule-identical to the full run.
    - prov: script
  -
    - text: `loss_curve[i]` is the pre-update loss of step i+1, averaged over 8 micro-batches × 4 × 1024 = 32,768 tokens.
    - prov: script
  -
    - text: `Shards` uses its own numpy rng seeded by `--seed`, independent of torch, so same seed means the same batches in every arm.
    - prov: script
  -
    - text: s and b must be initialised without torch RNG calls, and the counted prior must come from a fixed shard slice, not `train.rng`, or pairing breaks.
    - prov: belief
  -
    - text: Unpaired Welch with 3+3 seeds resolves 0.005 only if the per-arm sd is below 0.0016; paired with 3 seeds needs sd_d below 0.00124; paired with 5 needs below 0.0030. (t₀.₉₉: df2 = 6.965, df4 = 3.747, df7 = 2.998.)
    - prov: belief
  -
    - text: Losses go through `log_softmax`, which is on autocast's float32 list, so the logged loss is fp32 (https://pytorch.org/docs/stable/amp.html).
    - prov: web
  -
    - text: PyTorch documents several CUDA backward ops as nondeterministic, so same-seed reruns may diverge (https://pytorch.org/docs/stable/notes/randomness.html).
    - prov: web
  -
    - text: `clip_grad_norm_` (clip at 1.0) covers all parameters, including any new bias.
    - prov: script
  -
    - text: Scale-invariant weights trained with decay grow their norm through tangential steps until decay balances it (https://arxiv.org/abs/2006.08419). AdamW's per-coordinate normalisation also leaves a radial component, so L3's "norm set only by input and decay" is incomplete.
    - prov: web
  -
    - text: Three baseline seeds already exist with loss curves (from the layer-0 gradient measurement). Computing their across-seed sd of `mean(loss_curve[29:45])` is free.
    - prov: measured

open:
  -
    - q: What is the paired A/A floor: the sd of the `mean(loss_curve[29:45])` difference between two same-seed baseline reruns?
    - build: True
    - decides: 3 paired seeds vs 5–8 paired seeds
  -
    - q: What is the across-seed sd of the 15-step mean in the existing 3 baseline curves?
    - build: True
    - decides: welch kill vs paired kill
  -
    - q: Does the A/A floor grow by step 300 enough to make a later window worse than 30–45?
    - build: True
    - decides: kill window 30–45 vs 200–300
  -
    - q: Is the baseline's pre-clip grad norm above 1 in early steps, so that the bias gradient changes clipping?
    - build: True
    - decides: include s and b in clipping vs exclude them
  -
    - q: Will the harness accept an in-script train-slice probe?
    - build: True
    - decides: in-script probe vs `loss_curve` only
  -
    - q: What is the across-seed sd of the exam val loss at 1525 steps?
    - build: False
    - decides: (sets how many seeds the c7 claim needs)

offshoots:
  - s61fe702fbf
  - s20536be749
  - sbf9361031b
  - s22224e178f
  - s931cf77592

### New ideas born here
  - s06086e8ac4
  - s11a5cd9463
  - s6f95e7ea09
  - s1e1da9841a
  - s3d58dfadc3
  - s9490dc191f

### Pre-mortem objections
  -
    - text: The idea says "RMS-normalized rows" with a zero-init log-scale, so the scale starts at exp(0)=1. An RMS-normalized row has L2 norm √384≈19.6, and the final `self.norm` output also has norm ≈19.6. Initial logits therefore have std ≈19.6, compared with ≈0.4 in the baseline (rows at std 0.02). Step-1 l
    - severity: high
    - cls: impl_slip
    - if_true: - metric: learn_z
      - op: below
      - threshold: 0.0
  -
    - text: Normalizing the head multiplies AdamW's effective head learning rate by about 1/rms(row)≈50×. Adam's sign-like step is ~3e-3 per element on rows whose elements are ~0.02, so each step rotates a row's output direction by ~15%. Under the tie, the input lookup sees the same rotation. Weight decay shrin
    - severity: high
    - cls: hidden_cost
    - if_true: - metric: trend
      - op: above
      - threshold: 0.0
  -
    - text: A zero-init bias under AdamW at lr 3e-3 with this schedule can move at most about Σlr ≈ 3e-3 × ~1300 effective steps ≈ 4 nats over the whole run. The unigram prior spans roughly 10+ nats between rank-100 and rank-10k tokens. The bias therefore cannot hold the prior within budget. The "push rare toke
    - severity: high
    - cls: interaction_cancels
    - link: L2
    - if_true: - metric: learn_z
      - op: below
      - threshold: 1.0
  -
    - text: `build_optimizers` sends every 1-D parameter to AdamW with weight_decay=0.1. That contradicts L1's "kept out of weight decay". A counted log-unigram bias would be decayed toward 0 and lose about 30% of its spread by step 1525, so the rows would have to relearn that part of the prior late in training
    - severity: medium
    - cls: impl_slip
    - link: L1
    - if_true: - metric: long_z
      - op: below
      - threshold: 2.33
  -
    - text: Adding the bias or scale after the matmul (`logits * scale.exp() + bias`) creates extra (4096×50304) tensors. Autocast does not cover elementwise ops, so bf16 logits + fp32 bias is promoted to an 0.82 GB fp32 logits tensor that is kept for backward. This is the copy the inline comment warns about, a
    - severity: high
    - cls: breaks_constraint
    - if_true: - metric: step_ratio
      - op: above
      - threshold: 1.15
  -
    - text: If the frozen-prior or counted-bias variant computes counts in `GPT.__init__` (or at startup) using a glob like `*.bin`, it picks up `val.bin` whenever it is present. That trips C6. If counts are instead stored as a non-persistent buffer, or the prior is added only inside the `targets is not None` b
    - severity: high
    - cls: impl_slip
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: The headline gain comes from the prior, and that is early-only. Baseline is already at 7.06 at step 26, close to unigram entropy, and the probe's per-10-step gain halves after step 10. A model that gets the prior for free leads at steps 1–100, then the baseline catches up as its rows encode frequenc
    - severity: high
    - cls: early_only
    - link: L5
    - if_true: - metric: trend
      - op: above
      - threshold: 0.0
  -
    - text: The premise that "decay shrinks rare rows to zero" is weak at this budget. AdamW decoupled decay gives a total factor of about exp(−3e-4 × ~1300) ≈ 0.68 on every row equally, not rare-specific. The input side renormalizes anyway: block 0's `n1` RMSNorm sees the 4-stream read of the raw embedding. La
    - severity: medium
    - cls: below_noise
    - link: L4
    - if_true: - metric: learn_z
      - op: below
      - threshold: 1.0
  -
    - text: The output gradient's radial component goes away, but its tangential component still encodes frequency. With exp(s_v) fixed near 1 early, the only way to lower a rare token's logit is to rotate ê_v against typical h. Row direction then correlates with frequency instead of row norm, and the "clean in
    - severity: medium
    - cls: interaction_cancels
    - link: L3
  -
    - text: Even if the mechanism works, the predicted 0.005–0.03 nats at the end sits mostly at or below what C7 can detect. The test needs >0.005 above the pooled mean at p<0.01 with a one-sided Welch test over 3 seeds. With seed std around 0.005–0.01, roughly a 0.015+ effect is required. The bias-only varian
    - severity: medium
    - cls: below_noise
    - if_true: - metric: long_z
      - op: below
      - threshold: 2.33
  -
    - text: Rare-token loss (12% of tokens, 11.6 nats) is the stated target, but those tokens are mostly not predictable from context at 50M tokens. A good unigram prior lowers their loss only toward their unigram cost. The baseline reaches that cost too by mid-run, so the exam (the mean over all tokens) moves 
    - severity: low
    - cls: metric_blind
    - if_true: - metric: long_z
      - op: below
      - threshold: 1.0
  -
    - text: The 100.6k new scalars are counted as non-embedding by `param_counts`, giving 21.34M, about +0.5%, well inside C3. The extra per-microbatch rms_norm of the 19.3M-element table, forward and backward, ×8 accumulation steps, is small next to the 79-GFLOP head matmul. This is not where it fails
    - severity: low
    - cls: hidden_cost
    - if_true: - metric: param_ratio
      - op: above
      - threshold: 1.1

Pre-mortem verdict: Implemented as written, with a zero-init bias and RMS-normalized rows at unit scale, the head starts about 50× too hot and keeps a ~50× amplified effective learning rate. The zero-init bias cannot travel the ~10 nats of prior within the AdamW budget, so frequency is pushed back into row directions. 

## Lower mixing logit init (less identity)  (part State across depth: four-stream Sinkhorn residual mixing; prior: None)

### Chain and facts

chain:
  -
    - id: L1
    - text: Lowering the `mix_logits` init from 6 to 3, 2 or 1 makes more of each stream flow into the others at every mixing stage.
    - quantity: off-diagonal mass per stage, 3·b, read from the Sinkhorn output at step 0 and at the end
    - baseline: 0.74% at step 0; about 4.8% at the end from weight decay alone (belief)
    - expect: about 13% at init 3, 29% at init 2 and 52% at init 1 at step 0, growing over the run as decay shrinks the logits
    - breaks_if: init 6 already ends at 10% or more, or init 2 logits climb back above 4 (the optimiser re-sharpens to identity).
  -
    - id: L2
    - text: Read weights leave uniform, so the mixing matrix starts to affect the loss at all.
    - quantity: max abs(softmax(read_logits) − 0.25) over the 24 stages
    - baseline: 0 at init; unmeasured later
    - expect: above 0.05 by step 100, driven by Muon
    - breaks_if: it stays below 0.02 for the whole run, so the mixing has no effect on the loss at any init.
  -
    - id: L3
    - text: More exchange shrinks the differences between streams.
    - quantity: spread ratio ‖x − mean‖/‖mean‖ at block 11
    - baseline: unmeasured; it should be small because the write asymmetry is only 2%
    - expect: init 2 or lower gives a ratio at least 10x below init 6
    - breaks_if: the ratio is higher at init 2 than at init 6 (exchange creates differences instead of erasing them).
  -
    - id: L4
    - text: Smaller spread removes per-stream dominance but leaves the mean stream unchanged.
    - quantity: max-stream RMS / mean-stream RMS at block 11; mean-stream RMS growth
    - baseline: ×7.8 RMS growth in the single-stream baseline; this design is unmeasured
    - expect: the dominance ratio moves toward 1.0 at low init; mean-stream RMS growth changes by less than 3%
    - breaks_if: mean-stream RMS growth changes by more than 10% (column sums are not 1, or a precision artefact).
  -
    - id: L5
    - text: Less stream individuality changes the exam loss.
    - quantity: exam val loss over 3 seeds
    - baseline: pooled baseline mean (value not given)
    - expect: within about ±0.005 of init 6, drifting toward single-stream behaviour; a win larger than 0.005 is unlikely
    - breaks_if: init 2 or 3 beats init 6 by more than 0.005 at p < 0.01.

weakest:
  L2

qa:
  -
    - q: Does the loss depend on `mix_logits` at all while `read_logits` are uniform?
    - a: No. Sinkhorn ends with a column normalisation, so the column sums are exactly 1 in fp32 and the stream mean is invariant to M. Every read with uniform weights and the final `x.mean(dim=2)` read only the mean. So the forward pass is independent of M, and the gradient to `mix_logits` is zero up to bf16 rounding until the read logits move.
    - decides: rank the sweep by 45-step smoke loss vs rank it by a stream diagnostic or by full runs.
  -
    - q: Which way does a lower temperature move the differences between streams?
    - a: At init, M = aI + b(J − I), with eigenvalue 1 on the mean and a − b on every difference direction. - The per-stage factor a − b is 0.990 at init 6, 0.827 at init 3, 0.615 at init 2 and 0.30 at init 1. - Over 24 stages this compounds to 0.79, 0.010, 8.6e-6 and about 3e-13. - So inits of 2 or lower wipe out stream differences within a few blocks. The model then collapses to a single residual stream, which is the opposite of the stated mechanism.
    - decides: sweep {1, 2, 3, 6} vs sweep only {3, 4, 6} or switch to a non-contracting mix.
  -
    - q: Can streams 2 and 3 ever differ?
    - a: No (belief, from symmetry). - Both streams start as the same embedding copy and have identical write logits (−0.01 in both stages) and identical read logits. - The mix logits are symmetric under swapping 2 and 3, so the forward pass is equivariant under that swap. - Their gradients and updates are therefore identical, and they stay clones for the whole run. The design is effectively 3 streams, and no temperature fixes this.
    - decides: keep the current `write_logits` init vs give all four streams distinct read and write inits.
  -
    - q: Which optimiser updates the stream logits, and what does weight decay do to them?
    - a: Placement follows from the `ndim` filter in `build_optimizers`: - **`mix_logits` (2,4,4):** 3-D, so it goes to AdamW at 3e-3 with weight decay 0.1. Decay alone multiplies the diagonal by exp(−0.1·3e-3·~1281 effective steps) ≈ 0.68. That takes init 6 to about 4.1 (1.6% off-diagonal) and init 2 to about 1.36 (14.5% off-diagonal), so decay is already a hidden temperature schedule. - **`read_logits` and `write_logits` (2,4):** 2-D, so they go to Muon at lr 0.02 with weight decay 0.1. Newton–Schulz rescales the update to unit spectral norm, so even tiny gradients produce updates of about 0.01 per e
    - decides: leave the stream logits in the current groups vs a separate AdamW group with weight decay 0.
  -
    - q: Can a 45-step smoke run rank the inits?
    - a: No. - Warmup lasts 30 steps, so by step 45 the read logits have moved only about 0.2. The effect of M is then second order (read deviation × stream spread) and is swamped by rounding and seed noise. - C7 asks for a gap above 0.005 nats at p < 0.01, which only full runs can show. - A same-seed smoke sweep is useful only as a sanity check that the loss curves are nearly identical.
    - decides: smoke sweep vs going straight to full-length runs of two values.
  -
    - q: Does the 4-stream design beat the single-stream baseline at all?
    - a: Unknown; nothing in the brief measures it. At 1B scale, static ×4 hyper-connections gave a small gain over the baseline, and dynamic ×4 did better (https://arxiv.org/pdf/2409.19606). Because low temperature converges to single-stream behaviour (A2), a neutral 4-stream result implies a neutral sweep.
    - decides: spend runs on temperature vs first ablate to n = 1.
  -
    - q: Does the mixing push the residual through bf16?
    - a: Belief, not run. - Under autocast the einsum lowers to bmm, which runs in bf16. The carried residual is therefore rounded to bf16 at all 24 stages, even though the write addition promotes back to fp32. - After the bf16 cast, column sums are off by about 4–5e-4 at both init 6 and init 2. That shrinks the mean by roughly 1% over depth. Temperature barely changes this, but it adds noise to any comparison.
    - decides: compute the mix in fp32 with autocast disabled vs leave it as is.
  -
    - q: Can the final readout use stream differences?
    - a: No. `x.mean(dim=2)` is invariant to M and to how the signal is spread across streams. Only the 24 intermediate reads can exploit differences, which caps what any mixing temperature can buy.
    - decides: keep the mean readout vs a learned softmax final read.
  -
    - q: Where does the literature say Birkhoff-constrained mixing ends up, and has a lower init been tested?
    - a: Neither paper tested a lower init. - sHC reports that mHC matrices collapse toward identity and that non-negativity blocks subtractive mixing (https://arxiv.org/pdf/2603.20896). - The stream-collapse paper reports that the mix stays near identity while one stream dominates, and that a symmetry-breaking init helps (https://arxiv.org/pdf/2606.03483).
    - decides: lower init vs a symmetry-breaking init.
  -
    - q: Does the 20-iteration Sinkhorn cost matter for step time, and does the init change it?
    - a: The init does not change the cost. - Each optimizer step runs about 40 tiny kernels per stage × 24 stages × 8 micro-batches forward, plus the backward, although M depends only on the parameters. The wall-time share on the 4050 is unmeasured. - At init, M is a symmetric circulant matrix, so Sinkhorn is a no-op there.
    - decides: 20 iterations per forward vs an exact parameterisation or mixing at fewer stages.

known_tagged:
  -
    - text: `mix_logits` is a (2,4,4) Parameter initialised to 6·I. At init the diagonal is 0.9926 and each off-diagonal is 0.00246, not 99.75% identity as the variant states.
    - prov: script
  -
    - text: Diagonal and off-diagonal entries at the other inits: 0.870/0.043 at init 3, 0.711/0.096 at init 2, 0.475/0.175 at init 1.
    - prov: belief
  -
    - text: Sinkhorn ends with a column normalisation, so column sums are exactly 1 in fp32 and the stream mean is invariant to M.
    - prov: script
  -
    - text: All 4 streams start as the same copy of the embedding (`unsqueeze(2).expand`). The read logits are zeros, so every read is the mean, and the output is `x.mean(dim=2)`.
    - prov: script
  -
    - text: Write gains are 2·sigmoid(logit), which is 1.015 on the favoured stream and 0.995 elsewhere. The streams therefore differ by only about 2% of each write. Stage 0 favours stream 0, stage 1 favours stream 1, and streams 2 and 3 are never favoured.
    - prov: script
  -
    - text: `read_logits` and `write_logits` are 2-D, so they go to Muon at lr 0.02 with weight decay 0.1. `mix_logits` is 3-D, so it goes to AdamW at 3e-3 with weight decay 0.1.
    - prov: script
  -
    - text: The effective learning-rate sum over the run is about 1281 steps × lr: 1525 steps, 30 warmup, 458 cooldown.
    - prov: script
  -
    - text: Decoupled decay alone shrinks the diagonal mix logit by a factor of about 0.68 by the end, taking 6 to 4.1.
    - prov: belief
  -
    - text: The einsum runs in bf16 under autocast, so the carried residual is rounded to bf16 at every stage.
    - prov: belief
  -
    - text: The change adds 0 parameters and costs no extra memory or step time; C3 and C4 are untouched.
    - prov: script
  -
    - text: Single-stream baseline at step 100: RMS growth ×7.8, loss 6.1789, layer-0 attention gradient 0.24 against about 0.02 elsewhere.
    - prov: measured
  -
    - text: Static HC ×4 reaches an OLMo-1B eval loss of 2.791, against 2.781 for dynamic ×4. https://arxiv.org/pdf/2409.19606
    - prov: web
  -
    - text: On a 30M nanoGPT, alternative doubly-stochastic parameterisations reach only parity. https://arxiv.org/pdf/2604.02309
    - prov: web
  -
    - text: Version judged most likely to work: init 3 rather than 1 or 2, and only combined with distinct per-stream inits. On its own, expect a null result.
    - prov: belief
  -
    - text: What to log: spread ratio, max/mean stream RMS, read-weight deviation, the off-diagonal mass of M per stage at the end, and `torch.equal(x[:,:,2], x[:,:,3])`.
    - prov: belief

open:
  -
    - q: Does the 4-stream design beat n = 1 at all under this budget?
    - build: True
    - decides: temperature sweep vs dropping multi-stream
  -
    - q: How far do the read logits on Muon actually move by step 100 and by the end?
    - build: True
    - decides: leave them on muon vs move them to adamw with weight decay 0
  -
    - q: Does Newton–Schulz amplify noise-sized asymmetric gradients on the 2×4 write logits into real symmetry breaking?
    - build: True
    - decides: rely on the optimiser for symmetry breaking vs explicit distinct inits
  -
    - q: Is bf16 rounding of the residual at every stage costing loss relative to an fp32 mix?
    - build: True
    - decides: fp32 einsum under autocast disabled vs the current code
  -
    - q: What fraction of step time is the per-micro-batch Sinkhorn on the RTX 4050?
    - build: True
    - decides: 20 iterations vs an exact or sparse-stage mix
  -
    - q: What is the seed-to-seed spread of the exam loss for this script, and can 3 seeds resolve 0.005 nats?
    - build: False
    - decides: 3 vs 5 seeds
  -
    - q: Should weight decay on `mix_logits` be kept, since it already acts as a temperature schedule?
    - build: True
    - decides: weight decay 0.1 vs 0 on the stream logits

dropped_unsourced:
  1

### Deeper level: How far do the read logits on Muon actually move by step 100 and by the end?

qa:
  -
    - q: Does the size of the Muon update on `read_logits` depend on the size of its gradient at all?
    - a: No. `newton_schulz5` divides by the Frobenius norm and then orthogonalises, so every step's update has singular values of roughly 0.5–1.5 (https://kellerjordan.github.io/posts/muon/), and `scale = max(1, 2/4)**0.5 = 1`. The step is about `lr·m·(1–1.4)` in Frobenius norm whether the gradient is 1e-2 or 1e-6. Only a gradient with norm near the `1e-7` eps gets damped.
    - decides: L2 measured as logit movement vs L2 measured as a signal test such as update sign-consistency
  -
    - q: What is the lr-integrated movement budget at step 100 and at the end?
    - a: The run is 1525 steps, warm=30, cool_start=1067. Σ multiplier is 15.5+70 = 85.5 by step 100, giving Σlr = 1.71. Over the whole run it is 15.5+1037+229.5 = 1282, giving Σlr = 25.6. At 45 steps Σlr is only about 0.65, so a 45-step sweep sees at most a third of the step-100 movement.
    - decides: smoke length of 45 steps vs 100+ steps for any read-movement check
  -
    - q: What is the ceiling on read movement if the gradient direction were perfectly consistent?
    - a: The softmax gradient has zero row sum, so each update row is orthogonal to 1. The most a unit row can put on one entry is √(3/4) = 0.87. The ceiling is therefore about 1.5 logit units on the favoured entry by step 100, which gives a weight of about 0.70, i.e. max |w−0.25| ≈ 0.45. Under weight decay the equilibrium is |p| ≈ 0.87/0.1 ≈ 8.7, and 1−e^(−0.1·25.6) = 92% of it is reached by the end, which means near one-hot reads.
    - decides: bounded read (clamp or decay-only) vs free read
  -
    - q: Is the read gradient direction consistent (signal) or noise?
    - a: Unknown. At init the gradient is proportional to `<x_j − x̄, G>`, and the stream differences come from 2% write asymmetry plus bf16 rounding, so noise is plausible. If it is noise, Muon still takes full-size steps, and momentum 0.95 (persistence ≈ 20 steps) makes this a correlated random walk. My estimate (belief) is about 0.2–0.4 per entry by step 100, i.e. |w−0.25| ≈ 0.04–0.08, and an Ornstein–Uhlenbeck std of about 0.9 per entry by the end, i.e. |w−0.25| ≈ 0.15–0.2.
    - decides: keep reads on Muon vs move them to AdamW (Adam shrinks zero-mean gradients, Muon does not)
  -
    - q: Do `write_logits`, also 2-D on Muon, move just as fast and make the "2% asymmetry" an init-only fact?
    - a: Yes, the same arithmetic applies. By step 100 the write logits can move by up to about ±1.5, so gains of 1.015/0.995 can become roughly 1.6/0.4. Even a noise walk of ±0.3 gives gains of about 1.15/0.85. Stream differences will be set by Muon's write drift within tens of steps, not by the init.
    - decides: keep write logits on Muon vs freeze them or move them to AdamW to isolate the mixing effect
  -
    - q: Does read row 0 of block 0 (the attention read in the first block) receive any gradient?
    - a: Its input streams are always identical copies of the embedding, so any convex read gives the same output and its gradient is zero up to rounding. Inside NS5 that row is just a near-zero row of the 2×4 matrix; it is amplified at most about 3.44⁵ ≈ 490× relative, which stays negligible. It is a dead parameter and should be excluded from the 24-stage max (23 live stages).
    - decides: include vs exclude it from the L2 max; keep vs drop that parameter
  -
    - q: Is the stream state bf16, and does that hide small read changes?
    - a: Belief: yes. The `einsum` in `combine` runs under autocast, its inner bmm returns bf16, and `weights.to(x.dtype)` then rounds read weights to bf16. Near 0.25 the bf16 spacing is about 0.001–0.002, so deviations below 0.002 are invisible in the forward pass. The same rounding also turns 2% stream differences into partly quantised noise.
    - decides: measure read weights from the fp32 parameter vs from the forward; keep `combine` in bf16 vs force fp32
  -
    - q: Does Muon weight decay limit how far the reads and writes can go by the end?
    - a: Yes. `p.mul_(1 − lr·0.1)` gives an e-fold time of 500 full-lr steps. A consistent direction saturates at about 8.7 per entry (Q3), and noise settles near ±0.9. Decay does not prevent near one-hot reads if the signal is consistent.
    - decides: exclude stream gates from weight decay vs keep decay 0.1
  -
    - q: Does lowering the mix init change how far the reads move?
    - a: Not in magnitude, since Muon fixes the step size. It can change direction consistency: more exchange makes the streams more alike, so `x_j − x̄` shrinks toward rounding noise. Low inits are therefore more likely to show large but random read movement. Unknown until measured.
    - decides: one shared L2 gate vs a per-init L2 gate
  -
    - q: Can gradient clipping or the Muon eps stop read movement?
    - a: Clipping rescales the gradient, but NS5 then normalises it again, so clipping has no effect on direction. Only a read-gradient norm below about 1e-7 after clipping would shrink the step. That seems unlikely but is unmeasured, so log `‖grad(read_logits)‖` per stage.
    - decides: logging the read-gradient norm vs not

known_tagged:
  -
    - text: `read_logits` (2,4) and `write_logits` (2,4) are on Muon at lr 0.02, wd 0.1, momentum 0.95, nesterov; the NS5 scale factor is 1 for 2×4.
    - prov: script
  -
    - text: Muon step size is independent of gradient magnitude above about 1e-7 Frobenius norm; NS5 singular values are about 0.5–1.5, https://kellerjordan.github.io/posts/muon/
    - prov: web
  -
    - text: Σlr: 1.71 by step 100, 0.65 by step 45, 25.6 over 1525 steps (warm 30, cooldown from 1067).
    - prov: script
  -
    - text: Ceiling on a single read logit: about 1.5 by step 100 (|w−0.25| ≤ 0.45), and about 8 by the end with decay (equilibrium 8.7).
    - prov: belief
  -
    - text: Noise-only random walk: about 0.2–0.4 per entry by step 100 (|w−0.25| ≈ 0.04–0.08) and about ±0.9 by the end.
    - prov: belief
  -
    - text: Read row 0 of block 0 is dead: its streams are identical forever, so its gradient is about 0. Exclude it from L2.
    - prov: script
  -
    - text: Stream state is bf16 after the first `combine`. Read weights near 0.25 are quantised to about 0.001–0.002 steps in the forward.
    - prov: belief
  -
    - text: Step-0 off-diagonal mass 3b: 0.74% at init 6, 13.0% at 3, 28.9% at 2, 52.5% at 1. A uniform-row init is already doubly stochastic.
    - prov: script
  -
    - text: `mix_logits` on AdamW moves at most about 3.84 logit units over the run (Σlr 1282 × 3e-3); decay alone takes a diagonal of 6 to about 4.1.
    - prov: script
  -
    - text: What to log every 10 steps: fp32 `softmax(read_logits)` and `2·sigmoid(write_logits)` per stage, read-gradient norm, and the cosine between successive read updates.
    - prov: belief

open:
  -
    - q: Is the read update direction consistent across steps (cosine of successive Muon updates above 0.5) or a noise walk?
    - build: True
    - decides: reads on muon vs on adamw
  -
    - q: What is the read-gradient Frobenius norm per stage relative to the 1e-7 eps?
    - build: False
    - decides: logging only vs eps change
  -
    - q: How far do write gains actually drift by step 100 under Muon?
    - build: True
    - decides: l3 baseline taken at init vs measured at step 100
  -
    - q: Does bf16 stream storage erase stream differences below about 0.4%?
    - build: True
    - decides: bf16 `combine` vs fp32 `combine`
  -
    - q: Do read weights end near one-hot (
    - build: True
    - decides: decay on gates vs clamped or no-decay gates
  -
    - q: Does lower mix init make read movement more random?
    - build: True
    - decides: shared vs per-init l2 gate
  -
    - q: Do init-2 mix logits re-sharpen above 4?
    - build: True
    - decides: init 2 vs init 3

offshoots:
  - s400fda03d2
  - sfcab6002c5
  - s7bff612812
  - s807fb4c4e5
  - sca41315b3a
  - s998a036f1c

### New ideas born here
  - s32eadedb69
  - s2391ae38a8
  - sac31aaa55d
  - s76b0cb73ef
  - sf876630c85
  - s0942f451de

### Pre-mortem objections
  -
    - text: The mechanism's premise is wrong. Every read already sees all four streams through `softmax(read_logits)` over dim 2, so differences created by the writes reach every later read with no exchange at all. A doubly-stochastic mix acts on the deviation subspace (x − mean) only as a contraction. Its eige
    - severity: high
    - cls: interaction_cancels
    - link: L3
    - if_true: - metric: long_z
      - op: below
      - threshold: 2.33
  -
    - text: The loss is blind to M while reads are near uniform. The final readout is `x.mean(dim=2)`, and the last Sinkhorn step normalises columns to exactly 1 in fp32, so the mean stream is invariant to mixing. With `read_logits = 0`, the gradient into `mix_logits` is zero up to bf16 rounding. The loss depen
    - severity: high
    - cls: below_noise
    - link: L2
    - if_true: - metric: learn_z
      - op: below
      - threshold: 1.0
  -
    - text: Weight decay moves every arm toward full averaging. AdamW decay of 0.1 × 3e-3 over about 1280 effective full-lr steps shrinks each logit by about 0.68x. Init 6 ends near 4.1 (4.8% off-diagonal), init 2 near 1.36 (about 45%), and init 1 near 0.68 (about 66%, close to J/4 flat averaging). The low-init
    - severity: medium
    - cls: early_only
    - link: L1
    - if_true: - metric: trend
      - op: below
      - threshold: 0.0
  -
    - text: The 45-step smoke sweep cannot rank the inits. `--smoke-steps 45` sets total_steps = 45, so warmup is 1 step and cooldown starts at step 31, which is a different schedule from the full run. By step 45, Muon has moved `read_logits` at most about 0.45, and the 2% write asymmetry has produced almost no
    - severity: medium
    - cls: early_only
    - if_true: - metric: learn_z
      - op: below
      - threshold: 1.0
  -
    - text: Implementation slip in parameter routing: `read_logits` and `write_logits` are 2-D (2×4), so `build_optimizers` sends them to Muon at lr 0.02, while `mix_logits` (3-D) goes to AdamW at 3e-3.
    - severity: medium
  -
    - text: Muon's Newton–Schulz step normalises noise into full-size updates of about 0.01 per entry per step. With decay, the logits equilibrate near ±5, so write gains `2*sigmoid` can swing anywhere in [0, 2].
    - severity: medium
  -
    - text: The orthogonalisation also forces the attention-stage and MLP-stage rows' updates to be mutually orthogonal, which is an arbitrary coupling.
    - severity: medium
  -
    - text: Read/write dynamics run about 3x faster than the mix logits and dominate both stream behaviour and seed variance, so the mix init is a second-order knob.
    - severity: medium
  -
    - text: Cheapest check: log `write_logits` and the read softmax at steps 100, 500 and 1500 across 3 seeds.
    - severity: medium
  -
    - severity: medium
    - cls: impl_slip
    - link: L2
    - if_true: - metric: long_z
      - op: below
      - threshold: 2.33
  -
    - text: The residual runs in bf16. Under autocast, `torch.einsum` dispatches to bmm in bf16, so from block 0's first `combine` onward the 4-stream residual is bf16; the baseline keeps an fp32 residual because fp32 x plus a bf16 branch promotes. At RMS of about 10 in late layers, updates below about 0.04 are
    - severity: medium
    - cls: impl_slip
    - link: L4
    - if_true: - metric: long_z
      - op: below
      - threshold: 0.0
  -
    - text: The target metric cannot move. Mean-stream RMS is invariant under column-stochastic mixing (L4 concedes this). Total RMS includes only the roughly 2% spread, so removing that spread changes the ×7.8 growth by less than 0.1%. The growth comes from the branch outputs and the mixing never touches it.
    - severity: low
    - cls: metric_blind
    - link: L4

Pre-mortem verdict: On a doubly-stochastic, swap-symmetric mix, a lower init only contracts stream differences (about 0.6 survival per stage at init 2), so the model collapses to the single-stream baseline while still paying the bf16 residual and Sinkhorn overhead, and it lands within noise of the baseline or slightly 

## Slow batch-level gain on MLP write  (part Per-token computation: SwiGLU MLP; prior: new)
**Slow automatic gain control on the MLP write** (leap, card research, p_success 0.15)
- mechanism: Shared structure: a radio receiver chain has amplifier stages whose output envelope is measured over a window far longer than the signal, and a feedback gain keeps each stage inside the next stage's linear range; our chain is 12 blocks writing into a shared stream, the "signal" is the token-to-token variation of a write, the "envelope" is the batch-level RMS, and the stuck measurement is an MLP write RMS that drifts to 10.9 because magnitude can only be set by inflating three Muon-trained matrices against weight decay. Instantiation: the MLP keeps a detached running RMS of its own output (mome
- prediction: MLP output RMS stays within 2x of its initial per-layer level instead of growing to 10.9; matrix weight norms grow less; step time within 1%; parameters +12 and +12 buffers.
- test: 45 matched steps against the baseline curve; log per-layer write RMS and exp(a) every 10 steps; a 3-step smoke to confirm peak VRAM is unchanged.
- breaks: the magnitude of a sublayer's write into the residual is decided by the norms of its weight matrices.
- keeps: SwiGLU with its three matrices, widths, Muon on the matrices, the scaled init, the 4-stream combine.
- kill: mean loss over steps 40 to 45 more than 0.03 nats above the baseline's at the same steps, or any layer's gain exp(a)/ema outside [0.02, 50].
- target: residual RMS growth, down; update-norm spread, down.
- scope: validation loss at 1024
- source: own

### Chain and facts

chain:
  -
    - id: L1
    - text: Scaling the MLP output by c = exp(a)/ema_rms (ema detached) makes each layer's MLP write RMS follow exp(a) rather than the matrix norms
    - quantity: per-layer MLP write RMS before `combine`, divided by exp(a)
    - baseline: unmeasured (the 1.403→10.893 column is the residual stream after each block, not the MLP write)
    - expect: ratio within [0.5, 2] once the EMA settles; overshoot during warmup of e^{r·τ} (r = raw growth rate per step, τ = EMA lag in steps)
    - breaks_if: ratio stays outside [0.5, 2] after step 20, or tracks raw
  -
    - id: L2
    - text: Bounded MLP writes → less residual-stream growth across depth
    - quantity: residual RMS after block 11 / after block 0
    - baseline: x7.8 (1.403→10.893 at step 100)
    - expect: down to roughly x3–x6, but only if MLP writes are most of the stream
    - breaks_if: growth stays ≥ x7 (attention writes dominate, and they are untouched).
  -
    - id: L3
    - text: Write scale no longer depends on
    - baseline: median 0.02009, max 0.03265; weight norms unmeasured
    - expect: MLP
    - breaks_if: MLP
  -
    - id: L4
    - text: Smaller stream growth and steadier updates → equal or lower loss
    - quantity: mean loss over steps 40–45 (same total-step schedule as the baseline); final exam val loss
    - baseline: about 6.6–6.7 at steps 40–45 (6.59 at step 51); final is the pooled baseline mean
    - expect: within +0.03 at steps 40–45; final in the band −0.005 to −0.03
    - breaks_if: more than +0.03 at steps 40–45, or the final value does not beat the pooled mean by more than 0.005 at p<0.01.

weakest:
  L4

qa:
  -
    - q: In this pre-norm stack, can absolute residual RMS affect loss at all, or only the ratio of each write to the accumulated stream?
    - a: Every read goes through an RMSNorm (`n1`, `n2`, final `norm`), so a global rescale of the stream does not change the logits. Only the relative share of each write matters: the share of late layers shrinks as the stream grows. Whether that shrinking costs loss at this scale is unknown. If the mechanism is real, the relative target is the one that acts on it.
    - decides: absolute target (write RMS = exp(a)) vs relative target (write RMS = exp(a)·ema of block-input stream RMS)
  -
    - q: Under AGC, will Muon grow the MLP matrices less (as predicted) or more?
    - a: Probably more. With a detached EMA, each step's gradient still contains a "make the output bigger" radial component. Muon applies fixed-size updates regardless of gradient scale, so ||W|| keeps growing while the EMA cancels the effect on the output. That is the scale-invariant norm-growth effect, with effective step size falling as ||W|| grows (https://arxiv.org/abs/1803.01814, https://arxiv.org/abs/1706.05350). Weight decay 0.1·lr is the only brake. This is unmeasured, so log ||W||.
    - decides: leave W free vs add a norm constraint or extra weight decay on the MLP matrices
  -
    - q: Can a move far enough under the shared AdamW group (lr 3e-3, wd 0.1)?
    - a: Probably not. Adam moves a by at most about lr per step: roughly 0.12 nats over 45 steps and about 4 nats over the whole 1525-step run. In the baseline, writes go from about 0.01 (init) to O(1) by step 100, about 4.6 nats. A gain-1 init would leave the MLP nearly silenced early while attention writes grow freely. Give a its own group with lr around 0.03–0.1 and no weight decay. Weight decay would also pull a toward 0, i.e. gain 1/ema.
    - decides: a in the existing "other" group vs its own group (higher lr, no weight decay)
  -
    - q: How should a and the EMA be initialised?
    - a: Estimated initial write RMS is about 0.01. Inputs are normalized to RMS 1; w_gate/w_up std 0.02 gives pre-activation std of about 0.39; silu·up gives about 0.077; w_down std 0.00408 over 1077 inputs gives about 0.010. So a = 0 means gain ≈ 100, which trips the idea's own kill bound (>50) at step 1. Use lazy init: a sentinel EMA value replaced via `torch.where` on the first training forward (no `.item()` sync), and a = log(target), where the target comes from Q3/Q11.
    - decides: a = 0 vs a = log(first measured write RMS); EMA analytic init vs lazy init on the first training forward
  -
    - q: What EMA momentum, and updated per micro-batch or per optimizer step?
    - a: Early Muon relative updates are large. For w_down: lr 0.02 × ||orthogonal update||_F ≈ √384 ≈ 19.6 gives 0.39 per step against an init norm of 2.6, about 15% per step. Growth at rate r with lag τ leaves a constant overshoot of e^{rτ}. A 12-step window (0.99 per micro-batch with 8 micro-batches per step) could overshoot several-fold during warmup. Use about 0.9 per micro-batch, or bias-corrected momentum that ramps up. "Slow" is safe only after the warmup phase.
    - decides: momentum 0.9 vs 0.99 per micro-batch; update every micro-batch vs once per step
  -
    - q: Does the exam call `model.eval()` or run under `no_grad`?
    - a: Unknown. Gate on both, and register the EMA as a persistent buffer so `model.pt` carries it. Otherwise the exam would update statistics on validation tokens and make results depend on batch order.
    - decides: gate EMA updates on `self.training` only vs on `self.training and torch.is_grad_enabled()`
  -
    - q: Should the current batch's statistic enter the scale used for that same batch?
    - a: Update-then-use computes the RMS over all 1024 positions, so future tokens leak a tiny amount into earlier positions' scale. That is a causal violation, even if small, and train and eval would differ. Use the previous EMA, then update.
    - decides: use the previous EMA, then update vs update, then use
  -
    - q: Does the idea's kill bound on gain, exp(a)/ema ∈ [0.02, 50], fire by construction?
    - a: Likely. Raw output scales roughly with the product of three matrix norms. By step 45, raw growth is somewhere between about 1.6x (random-walk updates) and about 45x (aligned updates). The baseline reaching residual RMS 1.4 by step 100 suggests the aligned end. With a pinned (Q3), the gain falls toward or below 0.02 with the output still healthy. The bound measures W growth, not failure.
    - decides: keep the absolute-gain kill vs redefine it as gain relative to its step-0 value, or as write RMS / exp(a)
  -
    - q: Are MLP writes or attention writes the bigger part of the stream growth?
    - a: Unmeasured. Layer 0 shows anomalous gradients in both sublayers (attention 0.29, MLP 0.23, z 7.0 and 5.8). Log per-sublayer write RMS in the baseline before building; if attention dominates, L2 cannot hold.
    - decides: AGC on the MLP only (as specified) vs MLP plus attention
  -
    - q: How should the 45-step test match the baseline?
    - a: `lr_mult` scales with total steps. At 45 steps, warmup is 1 step and cooldown starts at step 31. At 100 steps, warmup is 2 steps and cooldown starts at 70. A 45-step run is therefore not matched to the 100-step baseline curve. Run 100 steps (or rerun the baseline at 45) and compare steps 40–45.
    - decides: `--smoke-steps 45` vs a 100-step run read at steps 40–45
  -
    - q: Should the EMA track mean-square or RMS, and in what precision?
    - a: Track sqrt(ema(mean y²)) in fp32. It is the unbiased envelope and avoids bf16 rounding across 12 layers × 8 micro-batches. Compute exp(a)/ema in fp32 and cast the scalar to `y.dtype`.
    - decides: sqrt(ema(mean y²)) in fp32 vs ema(rms) in bf16

known_tagged:
  -
    - text: Plug point: `MLP.forward`, after `w_down`, before `Block.combine` applies `2*sigmoid(write_logits)` (range 0–2) per stream.
    - prov: script
  -
    - text: The MLP write scale is already reachable without the matrices: the `n2.weight` gain, `read_logits`, and `write_logits` (capped at 2x), so the BREAKS premise is only partly true.
    - prov: script
  -
    - text: The 1.403→10.893 column is "Residual stream RMS after each block", not the MLP output RMS that the idea's mechanism cites.
    - prov: measured
  -
    - text: Residual RMS increments per layer are nearly constant (about 1.3 early, about 0.65 late), i.e. roughly linear growth, which suggests correlated writes rather than random ones.
    - prov: measured
  -
    - text: A 0-dim `a` falls into the AdamW "other" group (lr 3e-3, wd 0.1) unless split out.
    - prov: script
  -
    - text: Full run is 1525 optimizer steps, warmup 30, cooldown from step 1067, 8 micro-batches of 4×1024 per step.
    - prov: script
  -
    - text: Muon normalizes gradient scale, so c rescales W gradients without changing Muon's step size; it does change the global `clip_grad_norm_` factor seen by the Adam parameters.
    - prov: script
  -
    - text: Initial MLP write RMS is about 0.01 (derivation in A4).
    - prov: belief
  -
    - text: Muon per-step ||ΔW||_F is about 0.39 for w_down and about 0.66 for w_gate/w_up at full lr, against init norms of 2.6 and 12.9.
    - prov: belief
  -
    - text: Buffers are excluded from `param_counts`; +12 parameters is negligible against C3.
    - prov: script
  -
    - text: Scale-invariant weights grow in norm under gradient training and their effective lr decays: https://arxiv.org/abs/1803.01814
    - prov: web
  -
    - text: Peri-LN (normalizing sublayer outputs) bounds activation variance; its per-token output normalization is the known technique this idea must stay distinct from: https://arxiv.org/pdf/2502.02732
    - prov: web
  -
    - text: Learned scalar branch gains (ReZero, LayerScale) are the known non-normalizing relatives: https://arxiv.org/abs/2003.04887, https://arxiv.org/abs/2103.17239
    - prov: web

open:
  -
    - q: What is the per-layer MLP write RMS vs attention write RMS in the baseline at steps 0, 45, 100 and 1525?
    - build: True
    - decides: mlp-only agc vs mlp plus attention
  -
    - q: Does `nano/exam.py` call `model.eval()` and/or `torch.no_grad()`?
    - build: True
    - decides: ema update gated on `self.training` vs on `self.training and grad enabled`
  -
    - q: Does residual growth across depth measurably cost loss at this scale, e.g. via a frozen-ratio control run?
    - build: True
    - decides: absolute target vs relative-to-stream target
  -
    - q: Do MLP weight norms grow faster under AGC with Muon (the ratchet)?
    - build: True
    - decides: w left free vs norm projection or extra weight decay
  -
    - q: Which lr for a lets it track write growth without oscillating against the EMA lag?
    - build: True
    - decides: a in the shared adam group vs its own group with lr 0.03–0.1
  -
    - q: Is the EMA momentum that is safe during warmup (about 0.9 per micro-batch) too noisy later at batch-level RMS?
    - build: True
    - decides: fixed momentum vs a momentum schedule
  -
    - q: Is the near-linear residual growth caused by a shared DC direction in the writes?
    - build: False
    - decides: n/a

dropped_unsourced:
  1

### Deeper level: What is the per-layer MLP write RMS vs attention write RMS in the baseline at steps 0, 45, 100 and 1525?

qa:
  -
    - q: What is the per-layer MLP write RMS vs attention write RMS at step 0?
    - a: From the init (calculated, not measured): the MLP write is about 0.011 in every layer. RMSNorm input gives gate and up entries of std 0.02·√384 = 0.392, silu(g)·u has RMS about 0.081, and w_down contributes 0.00408·√1077 = 0.134. The attention write is 0.080·rms(y) with v std 0.226. Attention near 0.82-std logits over up to 1024 positions puts it between about 0.0015 and 0.018, most likely 0.002–0.005. So at init the MLP write is roughly 3–7x the attention write. Both are large relative to the 0.02-RMS embedding stream.
    - decides: apply the gain control to MLP only vs to both sublayers
  -
    - q: What do the measured numbers bound for the combined write per block at step 100?
    - a: By the triangle inequality, each block's combined write RMS is at least its residual increment: at least 1.3 early and at least 0.65 late. If writes were orthogonal to the stream, they would be about √(2.7²−1.4²) ≈ 2.3 early and √(10.9²−10.25²) ≈ 3.7 late. The split between MLP and attention is unknown. Either way, the writes are about 60–300x their init size by step 100.
    - decides: initialise exp(a) at the init write level (about 0.011) vs at the expected trained level (about 0.3–2)
  -
    - q: How far can `a` move under the default AdamW group?
    - a: Adam moves a parameter by at most about lr per step. The summed lr multiplier is about 85.5 by step 100 (warmup 30) and about 1281 over 1525 steps. So |Δa| is at most about 0.26 nats by step 100 (1.29x) and about 3.8 nats over the whole run (47x). Weight decay on `a` would also pull exp(a) toward 1 at 3e-4 nats per step, which matters if a ≈ −4.5 (exp(a) = 0.011).
    - decides: `a` in the AdamW "other" group (lr 3e-3, wd 0.1) vs its own group with higher lr and no weight decay
  -
    - q: Once c divides by the batch RMS, what else can still set the MLP write magnitude?
    - a: The batch-level divide cancels both the matrix norms and the n2.weight scale. What remains is exp(a) (at most 1.3x by step 100, per A3) and 2·sigmoid(write_logits), which is capped at 2x. So by step 100 the MLP write would be at most about 2.6x exp(a)'s starting value, unlike attention, which stays free.
    - decides: let n2.weight and write_logits keep training vs freeze them, or treat exp(a) as the only gain
  -
    - q: Does the raw (pre-c) MLP output keep growing under the gain control?
    - a: Very likely yes, though unmeasured. Muon's step size does not depend on the gradient scale, so multiplying by c does not slow W's norm growth. Scale-invariant weights grow in norm under gradient training (https://arxiv.org/abs/1803.01814). If the raw output grows about 100x while exp(a) moves 1.3x, then exp(a)/ema falls below 0.02 and the gain-bound KILL fires whether or not the loss is fine.
    - decides: kill on the bound exp(a)/ema ∈ [0.02, 50] vs kill only on exp(a) and loss
  -
    - q: How fast does the raw write grow per step, and what does that require of the EMA?
    - a: The rate r is unmeasured. A 100x rise over 100 steps gives r ≈ 0.046 per step. With τ = 100 steps (0.99 per step), the warmup overshoot e^{rτ} is about 100x. With τ ≈ 10 steps it is about 1.6x. `forward` runs 8 times per optimizer step, so momentum 0.99 per micro-batch means τ ≈ 12.5 steps.
    - decides: EMA momentum 0.99 vs 0.9, and updating per micro-batch vs per optimizer step
  -
    - q: Does the exam's forward update the EMA buffer?
    - a: Unknown. The harness calls GPT(cfg).forward(idx), and whether it calls .eval() or no_grad is not shown. The buffer must be persistent so model.pt carries it. Any update during the exam would adapt to val data and make the result depend on order, so the update should be gated on both conditions.
    - decides: gate the update on `self.training` vs on `torch.is_grad_enabled()` (or both)
  -
    - q: How do MLP and attention write magnitudes scale with uniform weight-norm growth?
    - a: The MLP write is about cubic in matrix scale: silu(g) ≈ g/2 for small g, so silu(g)·u·W_down scales as gate × up × down. The attention write is quadratic (expand × wo). q and k are normalized, so their norms only change how peaked attention is. Uniform norm growth therefore inflates MLP writes fastest, which supports targeting the MLP first.
    - decides: one gain mechanism for both vs MLP-specific
  -
    - q: Is the per-stream write equal to the residual increment, or does the 4-stream mixing split it?
    - a: At init, Sinkhorn on exp(6·I) gives diagonal 403.4/406.4 ≈ 0.993 and off-diagonal ≈ 0.0025. The write weights are 2·sigmoid(≈0) ≈ 1. So the four streams start nearly identical and each receives the full branch output. Measuring the EMA on the branch output, before `combine`, is enough unless the mixing de-diagonalises later (unknown).
    - decides: EMA on the branch output vs on the per-stream weighted write
  -
    - q: What equilibrium weight norm do Muon and decoupled weight decay reach, and when?
    - a: The decay time constant is 1/(lr·wd) = 500 steps. For uncorrelated updates, the Frobenius equilibrium is about ‖U‖_F·√(lr/2wd) (rotational equilibrium, https://arxiv.org/abs/2305.17212). For w_down that is about 6.2, or 0.0096 per entry vs 0.0041 at init. For w_gate and w_up (Muon scale 1.675) it is about 10.4, or 0.016 per entry vs 0.02 at init. Correlated updates, which the near-linear stream growth suggests, push norms above this. The real trajectory is unknown.
    - decides: expect raw growth to stop by about step 500 (fixed exp(a) suffices) vs expect continued drift (needs a fast `a`)

known_tagged:
  -
    - text: Plug point: `MLP.forward` output, after `w_down`, before `Block.combine`
    - prov: script
  -
    - text: MLP shapes: w_gate and w_up are 1077×384, w_down is 384×1077; init std 0.02, and w_down 0.02/√24 = 0.00408
    - prov: script
  -
    - text: Muon shape scale is sqrt(max(1, rows/cols)): w_gate and w_up 1.675, w_down 1.0, expand 2.449, wo 1.0, compress 1.0
    - prov: script
  -
    - text: MLP write RMS at step 0 ≈ 0.011 in every layer (calculated from init)
    - prov: belief
  -
    - text: Attention write RMS at step 0 ≈ 0.002–0.005, bounded by 0.0015–0.018 (calculated)
    - prov: belief
  -
    - text: Embedding stream RMS at init = 0.02, so step-0 MLP writes are about half the stream
    - prov: script
  -
    - text: Combined write per block at step 100 is at least 1.3 early and at least 0.65 late (triangle inequality on 1.403→10.893)
    - prov: measured
  -
    - text: SDPA default scale is 1/√96, since q is 64 content + 32 rotary dims
    - prov: script
  -
    - text: `forward` runs 8 times per optimizer step (32768 / (4·1024)); an EMA updated in `forward` updates per micro-batch
    - prov: script
  -
    - text: Summed lr multiplier: about 85.5 to step 100, about 1281.5 over 1525 steps (warmup 30, cooldown from 1067)
    - prov: script
  -
    - text: Decoupled weight decay time constant 1/(lr·wd) = 500 Muon steps; https://arxiv.org/abs/2305.17212
    - prov: web
  -
    - text: The batch-RMS divide cancels n2.weight's scale on the MLP path
    - prov: belief
  -
    - text: Buffers must be `register_buffer(..., persistent=True)` so model.pt carries the EMA for the exam
    - prov: script
  -
    - text: Muon's NS5 output has singular values near 1 regardless of gradient scale, so c does not change the size of W's updates; https://kellerjordan.github.io/posts/muon/
    - prov: web
  -
    - text: Instrumenting the question needs forward hooks on `b.attn` and `b.mlp` that log RMS at steps 0, 45, 100, 1525 into metrics.json. No new imports are needed, so C5 and C6 are untouched
    - prov: belief

open:
  -
    - q: Measured MLP and attention write RMS per layer at steps 45, 100, 1525
    - build: True
    - decides: init exp(a) at about 0.011 vs at the trained level
  -
    - q: Raw write growth rate r over steps 0–45
    - build: True
    - decides: ema momentum 0.99 vs 0.9
  -
    - q: Whether the exam calls .eval() or no_grad
    - build: True
    - decides: gate on self.training vs on grad-enabled
  -
    - q: Whether raw MLP output keeps growing under c
    - build: True
    - decides: gain-bound kill vs exp(a)-only kill
  -
    - q: Whether the 4-stream mixing de-diagonalises later in training
    - build: False
    - decides: branch-level vs per-stream ema
  -
    - q: What quantity L3 targets (its text is garbled)
    - build: True
    - decides: which weight-norm and δw statistics to log

offshoots:
  - sc451c4f7ee
  - scd0c7327b7
  - s2f1e61759d
  - s13296a9883
  - sbda52275bc
  - s197be56886

### New ideas born here
  - s92664facb8
  - s58d8a01d0d
  - s5b0cbe54a5
  - s912b21b5a6
  - s18916319a9
  - s04e31b5178

### Pre-mortem objections
  -
    - text: Muon removes the cost L3 claims to free.** Each Muon step applies `lr*scale*NS5(g)`, a fixed-size orthogonalized update, no matter whether the loss "wants" larger weights. Matrix norms in both arms are set by the Muon/weight-decay equilibrium (`p.mul_(1 - 0.02*0.1)` per step, about e^-3 over 1525 st
    - severity: high
    - cls: interaction_cancels
    - link: L3
    - if_true: - metric: long_z
      - op: below
      - threshold: 1.0
  -
    - text: Fixing the residual growth does not change the loss.** Every consumer of the stream is behind an RMSNorm (`n1`, `n2`, final `norm`, plus `q_norm`/`k_norm`), so absolute stream scale only matters as a ratio between writers. The baseline grows MLP writes so that deep layers stay audible against the ac
    - severity: high
    - cls: below_noise
    - link: L4
    - if_true: - metric: long_z
      - op: below
      - threshold: 2.33
  -
    - text: Attention dominates the growth, so L2 fails.** Attention writes through `wo` and the Sinkhorn/`write_logits` combine are untouched. The 1.403→10.893 figure is the stream after each whole block, with the MLP share unmeasured. If attention writes carry most of the growth, the stream ratio stays at x7 
    - severity: medium
    - cls: interaction_cancels
    - link: L2
    - if_true: - metric: learn_z
      - op: below
      - threshold: 0.5
  -
    - text: Only helps early.** The x7.8 growth was measured at step 100, the high-LR transient. Over the 30% linear cooldown to lr=0 both arms settle, so any early edge from bounded writes shrinks by the end. The 40–45 step kill test cannot see this decay
    - severity: medium
    - cls: early_only
    - if_true: - metric: trend
      - op: below
      - threshold: 0.0
  -
    - text: Exam-time buffer handling.** The harness rebuilds `GPT(cfg)` and loads `model.pt`. If the EMA is registered with `persistent=False`, it loads at its init value (for example 1.0) instead of about 0.01, and every MLP write is off by about 75x, which is catastrophic. If the EMA update is not guarded by
    - severity: high
    - cls: impl_slip
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: Weight decay pulls the gain parameter toward the wrong value.** `a` is 0-d or 1-d, so `build_optimizers` puts it in the AdamW group with `weight_decay=0.1`. If `a` is initialized to log(initial write RMS), about log(0.013) ≈ -4.3 given the 0.02/√24 `w_down` init, decay pulls `a` toward 0, which is a
    - severity: medium
    - cls: impl_slip
    - if_true: - metric: learn_z
      - op: below
      - threshold: 0.0
  -
    - text: EMA lag is in micro-batches and its start is unspecified.** The momentum got truncated out of the spec ("mome…"), and `accum=8`, so the EMA updates 8 times per optimizer step. A momentum chosen "per step" lags 8x less than intended, and the batch-noise in c rises. An EMA initialized at 0 without bia
    - severity: medium
    - cls: impl_slip
    - link: L1
    - if_true: - metric: learn_z
      - op: below
      - threshold: 0.0
  -
    - text: In-place EMA update can break autograd.** If `ema` is used in `x/ema` (saved for backward) and then updated in place with `mul_/add_` in the same forward, backward raises a version-counter error. A builder who "fixes" this by computing c from the current batch's RMS before updating creates batch-dep
    - severity: low
    - cls: impl_slip
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: Feedback loop with a detached controller.** The gradient on the MLP matrices treats c as a constant, so the optimizer keeps pushing the output RMS up. τ steps later the EMA cancels it. Muon momentum (0.95) plus EMA lag form a second-order loop that can ring: the gain oscillates and so does the effec
    - severity: medium
    - cls: interaction_cancels
    - link: L1
    - if_true: - metric: learn_z
      - op: below
      - threshold: 0.0
  -
    - text: Global grad clip couples the MLP to everything else.** Gradients into the MLP matrices scale with c, which is ∝ 1/
    - severity: medium
    - cls: hidden_cost
    - if_true: - metric: long_z
      - op: below
      - threshold: 0.0
  -
    - text: Scale invariance plus weight decay raises the late effective LR.** With the output normalized, `w_up`/`w_down` behave like BatchNorm-plus-weight-decay weights: their norm shrinks, while Muon's fixed-size update stays the same, so the angular step grows over training. This partly cancels the cooldown
    - severity: low
    - cls: early_only
    - if_true: - metric: trend
      - op: below
      - threshold: 0.0
  -
    - text: Hidden host syncs.** Logging `exp(a)/ema` every 10 steps, or a builder writing `float(ema)` or `.item()` inside the forward, adds 12 layers × 8 micro-batches of host syncs per step. The 1% step-time prediction breaks, though it stays well under the 2x limit
    - severity: low
    - cls: hidden_cost
    - if_true: - metric: step_ratio
      - op: above
      - threshold: 1.05

Pre-mortem verdict: Under pre-norm RMSNorm reads and Muon's fixed-size orthogonalized updates, the AGC only re-parameterizes write magnitude into `exp(a)` that the network can relearn, so the final exam loss stays inside seed noise (no 0.005-nat, p<0.01 win); the only way it moves the metric visibly is downward, throug

## Depth-shaped attention/MLP stage layout  (part State across depth: layer count and block layout; prior: None)

### Chain and facts

chain:
  -
    - id: L1
    - text: Re-layout (blocks 0–2 = A,A,M; blocks 3–9 = A,M; blocks 10–11 = M) -> one net extra attention stage
    - quantity: params_nonembed, tok/s, peak_vram_gb
    - baseline: 21.24M; tok/s and peak unmeasured here
    - expect: +0.53M (to 21.77M, +2.5%); step time +2–5%; peak VRAM +0.05–0.15 GB without checkpointing
    - breaks_if: step time rises more than 10% or peak goes above 5.6 GB
  -
    - id: L2
    - text: Attention in blocks 10–11 is redundant, so removing it costs about nothing
    - quantity: loss rise when the attention branches of blocks 10–11 are zeroed on a trained baseline checkpoint; secondarily, attention entropy in layers 10–11
    - baseline: entropy 0.711 / 0.763 (ranked 8th and 7th of 12, not highest); ablation unmeasured
    - expect: ablation rise below 0.02 nats
    - breaks_if: ablation rise is 0.03 nats or more, or recall-token loss rises by more than 0.05
  -
    - id: L3
    - text: A second attention stage in blocks 0–2 adds useful early mixing (two-hop composition such as previous-token followed by copy, before any MLP)
    - quantity: recall-token loss, and the position gap (256–1024 minus 0–64)
    - baseline: recall 4.475 vs 6.704 for the rest; gap +0.459
    - expect: recall loss down 0.02–0.10 at matched steps; gap shrinks
    - breaks_if: recall loss and the gap are unchanged (within ±0.02) at step 200
  -
    - id: L4
    - text: The new early stages actually learn
    - quantity: grad norm of each added attention stage
    - baseline: layer 0 attention 0.24 (an anomaly at z 7), layers 1–5 0.020–0.024
    - expect: added stages at or above 0.02
    - breaks_if: added-stage grad norm is below 0.01 by step 100, meaning the stage is dead weight
  -
    - id: L5
    - text: Net effect on exam loss
    - quantity: exam val_loss after 1525 steps
    - baseline: pooled baseline mean (not given here)
    - expect: −0.000 to −0.015 nats
    - breaks_if: the difference is within ±0.005, or step-45 train loss is more than 0.03 above baseline

weakest:
  L2

qa:
  -
    - q: Does the measured entropy profile support the premise that deep attention is near-uniform?
    - a: No. Entropy is highest at layer 0 (0.957), then layers 1 (0.903) and 2 (0.854). Layers 10 and 11 are 0.711 and 0.763, ranked 8th and 7th. By the idea's own rule ("if the last two blocks are not the highest-entropy layers, drop the idea"), the 100-step data already says drop it. The rationale is inverted: extra attention would go into the *most* uniform layers. The layout may still be right for other reasons (sandwich ordering, below), so the gate should be a zero-ablation of deep attention on a trained baseline checkpoint, scored on train-shard windows, not entropy.
    - decides: run the idea's own entropy gate vs replace it with an ablation gate
  -
    - q: Is there prior evidence for "more attention at the bottom, more FFN at the top" that does not depend on the entropy argument?
    - a: Yes. Press, Smith and Levy's sandwich transformer put extra attention sublayers at the bottom and extra FFN sublayers at the top, with the same sublayer count. It improved WikiText-103 and enwik8 language modelling, but the gain did not transfer to translation (https://arxiv.org/abs/1911.03864; recalled, not fetched this session). PAR Transformer similarly kept quality with fewer attention layers, placed mostly early (https://arxiv.org/abs/2009.04534; recalled). The idea's +0.53M params is a confound, so a parameter-neutral reorder is the needed control.
    - decides: keep the 13/12 non-neutral layout vs a parameter-neutral 12/12 reorder as the primary build
  -
    - q: In blocks 0–2, should the stage order be A-A-M or A-M-A?
    - a: A-A-M follows the sandwich pattern and lets the second attention read the first one's output before any MLP. A-M-A interleaves, which gives A,M,A,A,M,A,A,M,A,A,M… and just shifts where the doubled attention falls. Build A-A-M; it is the faithful reading of "a second attention stage".
    - decides: A-A-M vs A-M-A
  -
    - q: How should per-stage stream logits be shaped so the optimizer routing does not change silently?
    - a: In the script, `read_logits` and `write_logits` have shape (2,4), so the `p.ndim == 2` rule sends them to **Muon**. `mix_logits` (2,4,4) is 3-D and goes to AdamW. If the builder splits them per stage into 1-D (4,) tensors, they silently move to AdamW at lr 3e-3. Keep shapes (S,4), (S,4,4) with S = number of stages in the block. Note that MLP-only blocks then have (1,4) rows, and Muon normalises them to a unit row; that is still Muon and acceptable, but it should be noted.
    - decides: per-block (S,4) / (S,4,4) tensors vs per-stage 1-D parameters
  -
    - q: Is checkpointing block 0 needed?
    - a: Probably not. Activations for one attention stage including the four-stream tensors (each (4,1024,4,384) bf16 = 12.6 MB) come to roughly 80–120 MB, and the net change is +1 stage, about +0.1 GB. Baseline peak is unmeasured here, but "micro_batch 8 spilled past 6 GB" suggests micro_batch 4 sits around 3–3.5 GB. Build without checkpointing, read peak_vram_gb from `--smoke-steps 20`, and add checkpointing only if peak goes above 5.3 GB. Checkpointing also recomputes the Sinkhorn loops and adds code surface under the tripwire.
    - decides: torch.utils.checkpoint on block 0 vs no checkpointing
  -
    - q: Do the deep attention stages carry copying/recall work that MLP-only blocks cannot do?
    - a: Unknown at 21M / 50M tokens. At large scale, deep attention layers are the most removable by input/output similarity (He et al., https://arxiv.org/abs/2406.15786; recalled). Here the recall gap is 2.23 nats and nothing shows which layers produce it. The zero-ablation of block 10 and block 11 attention, separately, on a trained baseline decides this.
    - decides: drop attention in both blocks 10 and 11 vs only block 11
  -
    - q: Can the step-45 kill threshold (+0.03) be resolved?
    - a: Total steps = 50e6 // 32768 = 1525, and warmup = 30 steps, so step 45 is just past warmup. Loss falls about 0.019 per step there (7.062 at step 26, 6.590 at step 51), so a one-step lag reads as +0.02. Per-batch noise is unmeasured. Use the mean loss over steps 40–50, averaged over the same seeds for both arms.
    - decides: a single-seed step-45 comparison vs a 3-seed mean over steps 40–50
  -
    - q: Is a −0.01 gain detectable under C7 with 3 seeds?
    - a: Likely not. With n=3 per arm (Welch df ≈ 4, one-sided t₀.₀₁ ≈ 3.75), beating the baseline by 0.005 plus a margin requires a seed std below about 0.0016 for a true effect of 0.01. Seed std at 50M tokens is unmeasured. Plan 5 seeds (t ≈ 2.9, which needs std below about 0.0027).
    - decides: 3 seeds vs 5+ seeds per arm
  -
    - q: How will the exam reconstruct the layout?
    - a: The exam calls GPT(cfg) built from the saved config. Whether it uses the candidate's Config class is unknown. Derive the layout in code (or read it with `getattr(cfg, "layout", DEFAULT)`) so an exam-side Config without the field still builds the same model. A string field with a default also works through the auto-generated argparse.
    - decides: a new Config field vs a layout derived in code from n_layer
  -
    - q: What write-stream init should the third stage get?
    - a: It barely matters. 2·sigmoid(0.03) = 1.015 vs 2·sigmoid(−0.01) = 0.995, and mixing starts near identity (off-diagonal about 0.0025 after Sinkhorn on exp(6) diagonals). The streams stay nearly identical early on. Use stream s mod 4 for consistency. KNOWN - Attention stage params: wq 221,184 + compress 49,152 + latent_norm 128 + expand 98,304 + rotary 12,288 + q/k norms 128 + wo 147,456 = 528,640; with the pre-norm and logits, 529,048 per stage [script] - Net change is +529,048 params, giving non-embedding 21.77M (+2.5%), well under the C3 cap of 23.36M. The idea's "+0.59M" is wrong [script] - F
    - decides: write to stream 2 vs reuse stream 0's pattern

### New ideas born here
  - s329c40ad95
  - sfe1079881a
  - s6f953dd1aa
  - s1026759133
  - sbee2f4a7cf

### Pre-mortem objections
  -
    - text: The idea's own gate has already failed. Its TEST says to drop the idea unless blocks 10–11 are the highest-entropy layers. The measured entropies are 0.711 and 0.763, which rank 7th and 8th. Layer 0 is the flattest at 0.957, followed by layer 1 at 0.903. So the re-layout adds attention where attenti
    - severity: high
    - cls: interaction_cancels
    - link: L2
    - if_true: - metric: long_z
      - op: above
      - threshold: -1.0
  -
    - text: Entropy at step 100 says nothing about which layers become redundant at step 1525. Copying and induction-style heads form later in training. In a 12-layer model these heads tend to sit in the later half, because they need a previous-token head below them. Recall tokens are 13% of targets with loss 4
    - severity: high
    - cls: early_only
    - link: L2
    - if_true: - metric: trend
      - op: above
      - threshold: 0.0
  -
    - text: The early gain fades while the deep loss persists. A second attention stage before any MLP in blocks 0–2 gives a cheap two-hop path that speeds up the step-45 and step-200 recall metrics. By 1525 steps the baseline learns the same composition across blocks 1–4. The extra early stage then adds nothin
    - severity: high
    - cls: early_only
    - link: L3
    - if_true: - metric: trend
      - op: above
      - threshold: 0.0
  -
    - text: The expected effect is smaller than the test can detect. The predicted gain is −0.000 to −0.015 nats. C7 requires more than 0.005 nats at p<0.01 with a one-sided Welch test over 3 seeds. With seed-to-seed standard deviation around 0.005–0.01 at 50M tokens, even a true −0.01 effect is rarely signific
    - severity: high
    - cls: below_noise
    - link: L5
    - if_true: - metric: long_z
      - op: above
      - threshold: -2.3
  -
    - text: The step-45 kill criterion cannot see the change. At step 45 the model is in the unigram/bigram phase: loss falls mostly through embeddings and MLPs, and attention barely matters. Moving attention stages around therefore changes step-45 loss by much less than 0.03. The KILL gate passes almost any la
    - severity: medium
    - cls: metric_blind
    - link: L5
    - if_true: - metric: learn_z
      - op: above
      - threshold: -1.0
  -
    - text: The per-stage logits are hard-coded to two stages, which invites a builder slip. `mix_logits` has shape (2,4,4), `read_logits` (2,4) and `write_logits` (2,4), and `combine(..., stage)` indexes them by 0 or 1. The asymmetric write init routes attention writes to stream 0 and MLP writes to stream 1. A
    - severity: medium
    - cls: impl_slip
    - link: L1
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: `read_logits` and `write_logits` are 2-D and are not named tok or head, so `build_optimizers` sends them to Muon. Muon orthogonalizes them as a single matrix of shape stages×4. Growing this matrix from 2×4 to 3×4 in blocks 0–2 changes the Newton–Schulz output for the existing rows. The orthogonal up
    - severity: medium
    - cls: interaction_cancels
    - link: L4
    - if_true: - metric: learn_z
      - op: above
      - threshold: -1.0
  -
    - text: The added early attention stages may learn little, while the anomaly at layer 0 is not a real signal. Layer 0 attention already has a grad-norm anomaly (0.29, z 7) and the flattest entropy (0.957). Attention on raw embeddings acts largely as bag-of-words averaging. A second attention stage reading t
    - severity: medium
    - cls: interaction_cancels
    - link: L4
    - if_true: - metric: learn_z
      - op: above
      - threshold: -1.0
  -
    - text: Checkpointing block 0 has a hidden step-time cost, and the cost estimate assumes plain residual blocks. Block 0 now holds two attention stages and three four-stream combines. Each combine runs a Sinkhorn mix and an einsum over a (B,T,4,384) tensor, both memory-bound, so at d=384 they cost about as m
    - severity: low
    - cls: hidden_cost
    - link: L1
    - if_true: - metric: step_ratio
      - op: above
      - threshold: 1.1
  -
    - text: The exam can silently disagree with the training run. If the layout is controlled by a new Config field, it is saved in model.pt, and the exam rebuilds the model with GPT(cfg). If instead it is controlled by a module-level constant or a CLI flag outside `asdict(Config())`, the exam builds a model wh
    - severity: medium
    - cls: impl_slip
    - link: L1
    - if_true: - metric: fidelity
      - op: below
      - threshold: 1.0
  -
    - text: Removing attention from the top two blocks also removes 2 of the 4 last stream mixes into the output. The output is the mean over 4 streams. Blocks 10–11 then only move MLP output into the streams through stage-1 writes. Attention-routed information then reaches the head only through the slower Sink
    - severity: low
    - cls: interaction_cancels
    - link: L2
    - if_true: - metric: long_z
      - op: above
      - threshold: -1.0

Pre-mortem verdict: The baseline's own measurements already refute the premise: the deepest layers are not the near-uniform ones (layer 0 is), so the idea removes useful late attention, which recall/induction behaviour needs by step 1525, to add attention where it is most redundant, and any small early gain fades into 

## Decay-proof per-head log-temperature and exp-gains  (part Sequence mixing: decoupled rotary dims and QK-norm; prior: known)
**Re-entangle: RoPE on 16 content dims, delete the decoupled key** (leap, card research, p_success 0.15)
- mechanism: Decoupled RoPE exists so the latent can be absorbed into the query at inference with a cache; this script never runs with a cache, so the 32 rotary q dims per head plus the shared rotary projection are a cost paid for nothing. Remove Attention.rotary and the rotary slice of wq, apply RoPE to 16 of the 64 content dims of q and k after q_norm and k_norm (rotation preserves the norm, so QK-norm still holds), with the theta-480 spectrum over 8 pairs; positional keys become per-head and content-dependent through the latent.
- prediction: Non-embedding parameters fall by 86,016 per layer, 1.03M total; SDPA head dim 96 to 64 cuts attention-score compute by a third, about 3 percent of the step; loss equal to or 0.005 below baseline, which counts because fewer parameters are always allowed.
- test: Parameter count by --build-only (20.21M expected); 3-step timing; 45-step matched smoke; the partial-RoPE paper's finding that 10 percent of dims suffice says 25 percent should not hurt.
- breaks: Position and content must live in disjoint channels because MLA's inference-time absorption requires it.
- keeps: Latent KV compression, QK-norm, 6 heads of 64, output projection, causal SDPA.
- kill: 45-step loss above baseline by 0.03, which would mean the full-rank x-to-rotary path, not the decoupling, is what the keys needed.
- target: validation loss only, with step time down
- scope: validation loss at 1024
- source: own; builds on https://arxiv.org/abs/2603.11611 and the open item in https://arxiv.org/pdf/2503.11486

### Chain and facts

chain:
  -
    - id: L1
    - text: Replace the q_norm/k_norm gains with exp(γ) and add a per-head exp(τ_eff,h) on q -> higher effective attention temperature in deep layers during training
    - quantity: per layer, exp(τ_h)·mean(g_q⊙g_k) for content, and exp(τ_h) for the rotary part
    - baseline: unmeasured (gains are initialised at 1; no checkpoint dump exists)
    - expect: up 10–50% in layers 6–11 by step 300; layers 0–2 flat or down
  -
    - id: L2
    - text: Higher temperature -> larger max attention logit in deep layers
    - quantity: max attention logit by layer
    - baseline: layers 7–11 = 19.75, 22.75, 20.12, 20.88, 16.62 at step 100
    - expect: up 5–30% against baseline at the same step (300)
    - breaks_if: τ rises but max logit does not move, which would mean the rotary part sets the max and τ only rescales the content part that does not
  -
    - id: L3
    - text: Larger deep logits -> deep attention sharpens earlier
    - quantity: attention entropy, layers 6–11
    - baseline: 0.698, 0.704, 0.705, 0.667, 0.711, 0.763 at step 100
    - expect: down 0.02–0.08 at steps 100–300
    - breaks_if: entropy is unchanged within seed spread
  -
    - id: L4
    - text: Sharper deep attention -> better copying/recall
    - quantity: recall-token loss (bigram already seen in the window)
    - baseline: 4.475 vs 6.704 for the rest
    - expect: recall loss down 0.02–0.10; non-recall within ±0.01
    - breaks_if: recall loss does not fall more than non-recall loss
  -
    - id: L5
    - text: -> exam validation loss at 50M tokens
    - quantity: exam val_loss
    - baseline: pooled baseline mean
    - expect: down 0.005–0.015
    - breaks_if: one-sided Welch over 3 or more seeds misses p < 0.01 at a 0.005 margin

weakest:
  L1

qa:
  -
    - q: Is the measured max logit of 22.75 (layer 8, step 100) produced by the RMS-normed content part or by the unnormalised rotary part?
    - a: By step 100 the content part cannot be the main source. The cumulative AdamW lr multiplier over steps 1–100 is about 15 + 70 = 85, so each gain element moved by at most about 85 × 3e-3 ≈ 0.26 (Adam's per-step move is roughly bounded by lr). That gives g ≤ ~1.26 and a perfectly aligned content logit ≤ 64·1.26²/√96 ≈ 10.4. Most of the 22.75 must come from qr·kr/√96. Those vectors come from wq and self.rotary with no norm, so their scale is set by Muon. The idea's "gains grew to 1.9" is therefore inferred, not observed. Scaling the whole q is the faithful build, but the builder must log the conte
    - decides: apply exp(τ) to the whole q (content + rotary) vs only to qc
  -
    - q: Is AdamW decay actually binding on the gains?
    - a: Pure decay over the run is a factor of exp(−3e-4 × ~1281 effective steps) ≈ 0.68. Adam moves each element by up to 3e-3 per step against a decay pull of 5.7e-4 at g = 1.9. The equilibrium needs a gradient signal-to-noise ratio of only m/√v = wd·g ≈ 0.19, so the decay binds only where that ratio is weak. Whether it does is unknown until the baseline model.pt gains are dumped.
    - decides: ship the bundle (exp(γ) + τ) vs τ alone with linear gains
  -
    - q: Does the baseline already have any per-head content temperature?
    - a: No. q_norm and k_norm are RMSNorm(64) applied on the last dim of (B,H,T,64), so the 64 gains are shared across all 6 heads. Each head's content logit scale is identical. Per-head τ is new expressiveness and may be the real active ingredient, more than the decay-proofing. Build per-head as specified, and treat "τ-only" as the main ablation.
    - decides: τ per head (6 per layer) vs one τ per layer
  -
    - q: How should a per-head scale go into SDPA?
    - a: The `scale` argument of `F.scaled_dot_product_attention` takes only a Python float. Pre-multiplying q keeps the fused flash/efficient kernel and costs one extra (4,6,1024,96) bf16 tensor (~4.7 MB per layer). Materialising 4×6×1024×1024 scores per layer would cost VRAM and step time against C4. Use the pre-multiply.
    - decides: pre-multiply q by exp(τ_eff)[None,:,None,None] with the default scale vs materialise QKᵀ with a tensor scale
  -
    - q: Which soft clip keeps τ = 0 exactly at 1/√96?
    - a: The sigmoid form maps τ = 0 to exactly 0, ranges over (−1, 2), and has slope 2/3 at 0, so τ learns about 2/3 as fast. A hard clamp has zero gradient outside the range, so a head can get stuck at the bound. A centred tanh does not give 0 at τ = 0. Use the sigmoid form. Weight decay on the raw τ then returns it to the unit scale, which is the intended fixed point.
    - decides: τ_eff = 3·σ(τ − ln 2) − 1 vs hard clamp(τ, −1, 2)
  -
    - q: Does exp(γ) change more than the decay fixed point?
    - a: Yes. Adam's update on γ is multiplicative, so the effective step on g is lr·g: about 1.9× faster at g = 1.9 and slower below 1. The decay pull at the same g also drops about 3× (wd·ln 1.9 = 0.064 vs wd·1.9 = 0.19). Any result therefore confounds "decay-proof" with "faster gain learning". A third arm is needed to attribute a win.
    - decides: bundle γ-reparam and τ in one arm vs add a control arm with linear gains + τ
  -
    - q: Can the 45-step smoke test discriminate at all?
    - a: With --smoke-steps 45 the schedule compresses: warmup 1 step, cooldown from step 31. Total multiplier is about 31 + 7 = 38, so γ and τ move ≤ ~0.11 and the temperature changes by ≤ ~8%. The 45-step check can catch breakage (the +0.02 kill) but cannot show the mechanism. L1–L3 need ≥300 steps with per-layer logit and entropy dumps.
    - decides: 45-step kill check vs a 300-step matched loss_curve comparison
  -
    - q: How exactly is "gain norm within 1.15 of 1.0" measured?
    - a: The L2 norm of the initial 64-vector is 8, so the criterion must mean RMS or mean. The quantity that multiplies the logit is Σ_d g_q,d·g_k,d·q̂_d·k̂_d, so mean(g_q⊙g_k) per layer is the right metric. A kill threshold of 1.15² ≈ 1.32 on the product is equivalent.
    - decides: kill on RMS(g) per norm vs on mean(g_q⊙g_k) per layer
  -
    - q: Is an upper bound of 2 safe?
    - a: Unknown. QK-norm is the current guard against logit blow-up. At 7.4×, content logits could reach about 7.4 × 6.5 ≈ 48, and the unbounded rotary part could exceed that. fp32 softmax inside the kernels will not overflow, but entropy collapse and loss spikes are possible. Keep 2 as the idea says, and log max τ_eff per layer and loss spikes.
    - decides: τ_max = 2 (7.4× logits) vs τ_max = ln 3 (3×)
  -
    - q: Is there prior evidence at this scale?
    - a: QKNorm with a learnable g gave +0.928 BLEU on small MT models (https://arxiv.org/pdf/2010.04245). SSMax reports faster pretraining loss reduction (https://arxiv.org/pdf/2501.19399). Unclipped learned per-token temperature drifted and hurt performance (https://arxiv.org/pdf/2310.12956). I know of no sub-100M LM ablation with MLA and decoupled RoPE. Clipping is supported by that evidence; the size of the effect is unknown.
    - decides: prioritise this over the other OPEN temperature variants vs not

known_tagged:
  -
    - text: q_norm and k_norm are RMSNorm(64), one pair per Attention, applied to (B,H,T,64). Their weights are shared across the 6 heads, giving 128 gains per layer and 1536 total
    - prov: script
  -
    - text: Every 1-D parameter, including norm gains, goes to AdamW with lr 3e-3 and weight_decay 0.1. Decay per step at peak is 3e-4
    - prov: script
  -
    - text: total_steps = int(50e6/32768) = 1525; warmup 30 steps; cooldown from step 1067; summed lr multiplier ≈ 1281
    - prov: script
  -
    - text: The SDPA default scale is 1/√96, because q and k are 64 content + 32 rotary dims
    - prov: script
  -
    - text: qr (from wq) and kr (from self.rotary, shared across heads) are not normalised. Their scale is set by Muon with lr 0.02 and wd 0.1, i.e. decay 2e-3 per step
    - prov: script
  -
    - text: Max attention logit at step 100 by layer: 7.81, 15.5, 13.5, 18.12, 15.5, 16.0, 17.38, 19.75, 22.75, 20.12, 20.88, 16.62
    - prov: measured
  -
    - text: Entropy is lowest at layer 9 (0.667) and highest at layer 0 (0.957). First-token mass is ≤ 0.006 everywhere, so there are no strong sinks
    - prov: measured
  -
    - text: Layers are indexed 0–11. The prediction's "layers 8 to 12" should read 7–11 or 8–11
    - prov: script
  -
    - text: `F.scaled_dot_product_attention(scale=)` accepts only a float, so per-head τ must be applied by pre-multiplying q
    - prov: belief
  -
    - text: Use τ_eff = 3·σ(τ − ln 2) − 1, which gives exactly 0 at τ = 0, range (−1, 2), slope 2/3
    - prov: belief
  -
    - text: Change only q_norm and k_norm, through a new class. Do not touch the global RMSNorm used by n1, n2, latent_norm and the final norm
    - prov: script
  -
    - text: Initialise γ = 0 and τ = 0, so step 0 is bit-identical in function to the baseline
    - prov: belief
  -
    - text: The parameter delta is +72 (12 layers × 6 heads); γ replaces weight one-for-one. This is well inside the C3 margin over 21.24M
    - prov: script
  -
    - text: The state_dict keys change (q_norm.weight → q_norm.gamma). The exam must build GPT from this file for the weights to load
    - prov: belief
  -
    - text: Learned per-token temperature without clipping drifts and degrades performance (https://arxiv.org/pdf/2310.12956)
    - prov: web
  -
    - text: Learnable QKNorm scale g gave an average +0.928 BLEU on 5 low-resource pairs (https://arxiv.org/pdf/2010.04245)
    - prov: web
  -
    - text: NoPE with QK-Norm ends at higher loss than runs with ≥10% RoPE. Here rotary is 32/96 ≈ 33% of q/k dims (https://arxiv.org/abs/2603.11611)
    - prov: web
  -
    - text: Watch per layer: τ_eff per head, mean(g_q⊙g_k), content and rotary max logits logged separately, entropy, and recall-token loss
    - prov: belief

open:
  -
    - q: What fraction of the deep-layer max logit comes from the rotary part vs the content part, at steps 100 and 1525?
    - build: True
    - decides: τ on all of q vs only on qc (or split τc/τr)
  -
    - q: What are the baseline full-run values of mean(g_q⊙g_k) per layer in model.pt?
    - build: True
    - decides: include the exp(γ) reparam vs τ-only
  -
    - q: Does the step-100 sharpness pattern hold at step 1525?
    - build: False
    - decides: τ in all layers vs only layers 6–11
  -
    - q: Is the step-to-step seed noise of loss_curve at step 45 below 0.02, i.e. is the kill threshold meaningful?
    - build: False
    - decides: 45-step kill vs 300-step kill
  -
    - q: Does the exam build GPT from this file so that renamed state_dict keys load?
    - build: True
    - decides: rename the parameter to gamma vs keep the key name weight and store the log value
  -
    - q: Does τ_max = 2 cause loss spikes once QK-norm's guard is relaxed?
    - build: True
    - decides: upper clip 2 vs ln 3
  -
    - q: Does the slope 2/3 of the sigmoid clip slow τ enough to hide the effect within 1525 steps?
    - build: True
    - decides: sigmoid clip vs a rescaled clip with slope 1 at 0

dropped_unsourced:
  2

### New ideas born here
  - s5c15bc950a
  - s32822ad5c4
  - s98742624a8
  - sa7a853416d
  - sb951f6724d
  - s977f6a937d

## Shifted-key one-hop induction  (part Sequence mixing: head count, head structure and output proje; prior: new)
**Shifted-key heads (one-hop induction)** (leap, card research, p_success 0.18)
- mechanism: For heads 3 to 5, the key at position j is the (post-RoPE) key of token j−1 while the value stays token j's; a query that matches "the token before me last time" directly reads "what followed it". Today this needs two layers (a previous-token head feeding an induction head); fusing it frees those layers for other circuits and should lower loss on repeated spans within 1024 tokens.
- prediction: Zero parameter change, step time within 1%; training loss separates from baseline after the induction bump (roughly step 150 to 300), gap of 0.02 to 0.04 nats at 50M tokens, largest on late positions.
- test: 45-step matched smoke run on seed 0; 3-step timing; unit check that query i never sees token i+1 (compare logits for a sequence truncated at i).
- breaks: The key a query scores and the value it reads belong to the same token.
- keeps: softmax causal attention, the latent KV path, QK-norm, RoPE on the rotary slice, the same wo and head count.
- kill: mean loss over steps 40 to 45 more than 0.02 nats above the baseline's at the same steps.
- target: recall-token loss, down; loss at late positions, down
- scope: validation loss at 1024
- source: own

### Chain and facts

chain:
  -
    - id: L1
    - text: Heads 3–5 use the key of token j−1 with the value of token j, so in one layer they can attend to positions whose predecessor matches the current token
    - quantity: share of attention mass that heads 3–5 put on "successor of an earlier copy of token i" positions, per layer
    - baseline: unmeasured
    - expect: up, from chance (<0.05) to >0.2 in at least one layer by step ~300
    - breaks_if: this share stays near chance in every layer through step 500.
  -
    - id: L2
    - text: One-hop induction lowers loss on repeated spans early in training
    - quantity: recall-token loss (targets whose bigram was already seen in the window)
    - baseline: 4.475 at step 100 (vs 6.704 for other tokens)
    - expect: down 0.15–0.3 nats against baseline at matched steps 200–1525
    - breaks_if: the gap at matched steps is under 0.05 nats.
  -
    - id: L3
    - text: Recall gains build up with context length
    - quantity: loss at 256–1024 minus loss at 0–64
    - baseline: +0.459 at step 100
    - expect: this difference shrinks against baseline by 0.02–0.1
    - breaks_if: the late-minus-early difference matches baseline within 0.01.
  -
    - id: L4
    - text: Taking 3 of 6 standard heads away in all 12 layers costs nothing on the other 87% of targets
    - quantity: non-recall token loss
    - baseline: 6.704 at step 100
    - expect: within ±0.01 of baseline
    - breaks_if: it rises by more than ~0.15 × the recall-token gain, which cancels the net effect.
  -
    - id: L5
    - text: Net effect on the exam
    - quantity: exam val loss at 50M tokens
    - baseline: pooled baseline mean (not in the measurements above)
    - expect: down 0.01–0.04
    - breaks_if: the gap is ≤0.005 or one-sided Welch p ≥ 0.01 over 3 seeds.

weakest:
  L4

qa:
  -
    - q: Is the predicted 0.02–0.04 nats arithmetically reachable through recall tokens alone?
    - a: Recall tokens are 13% of targets (measured). A 0.02 net gain needs about 0.15 nats on them, and 0.04 needs about 0.31 nats, assuming zero cost elsewhere. That is a large recall gain, so the build cannot also afford a 0.01 loss on the other 87% of targets. This argues for logging both groups from the first run.
    - decides: 3 shifted heads per layer vs 1 shifted head per layer (lower cost to L4)
  -
    - q: How should the shifted key at position 0, which has no predecessor, be filled?
    - a: `torch.roll` wraps token T−1's key into position 0. That puts future content into a key every query can see, so it is a leak and must not be used. The safe choice is a zero key: `cat([zeros_like(k[:,:,:1]), k[:,:,:-1]], dim=2)`. It gives logit 0, so query 0 attends only to j=0 and reads its own value (no NaN). Watch whether position 0 becomes a sink for heads 3–5; baseline first-token mass is 0.002–0.006.
    - decides: a zero key vs `torch.roll` (or a duplicate of token 0's key)
  -
    - q: Should the whole key be shifted (content plus rotary, post-RoPE), or only the content slice with rotary at j?
    - a: The idea specifies post-RoPE, so shift the whole key. The rotary score then sees relative offset i−j+1, and purely positional heads still work, offset by one. `kr` is a single head expanded to all heads, so the simplest faithful build shifts the final `k` tensor for heads 3:6 after the `torch.cat`. Both variants are causal.
    - decides: shift the final concatenated `k[:, 3:6]` vs shift `self.k_norm(k)` only and keep `kr` at j
  -
    - q: Can QK-norm logits get sharp enough to pick one matching token out of ~1000?
    - a: With norm weights at 1, the content logit is at most 8·8/√96 ≈ 6.5 (SDPA scale is 1/√96 because q has 96 dims). Selecting one position among ~1000 needs a gap of about ln(1000) ≈ 6.9. The baseline already grows its max logit to 7.8–22.75 by step 100 (measured), so the norm gains learn this quickly. Keep init 1 for fidelity.
    - decides: keep the `q_norm`/`k_norm` weight init at 1 vs raise it (e.g. 2)
  -
    - q: Does induction even form inside 50M tokens (1525 steps), and when?
    - a: Unknown for this recipe. Olsson et al. report the induction phase change in small models at roughly 2.5–5B tokens with large batches (https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). This setup has a different optimizer, batch and LR, so that does not transfer. At step 45 (1.47M tokens) neither model likely has induction, so the 45-step kill tests only harm, not benefit.
    - decides: trust the 45-step kill vs add a matched 300-step run before the 3-seed run
  -
    - q: Which layers should get shifted heads?
    - a: Unknown. The mechanism needs the key to reflect token j−1's identity, which is cleanest in early layers where the residual is close to the embedding. Later layers already mix in neighbour information through standard heads. Build it as specified, but log the L1 attention share per layer so a specialised version can be chosen from data.
    - decides: heads 3–5 in all 12 layers (as specified) vs only in layers 0–5
  -
    - q: Does it fit C3 and C4?
    - a: It adds no parameters, so non-embedding stays at 21.24M. Each layer holds an extra (4,3,1024,96) bf16 tensor, about 2.4 MB, roughly 28 MB across 12 layers; that is negligible against 5.6 GB. SDPA keeps `is_causal=True`, so the fused kernel path is unchanged. A `cat` of a slice should stay within ~1% step time, but measure it in the 3-step timing check.
    - decides: `cat`-based shift inside `Attention.forward` vs a precomputed index gather
  -
    - q: Can the exam harness rebuild the shifted model from `model.pt`?
    - a: Unknown how `nano/exam.py` maps the saved config onto `Config`. Hardcoding (heads `n_head//2:` shifted) reproduces the model whatever the harness does with extra keys. A field with a default equal to the new behaviour is also safe if the exam instantiates this file's `Config`.
    - decides: hardcode the shift vs add a Config field (e.g. `shift_heads: int = 3`)
  -
    - q: Does the layer-0 gradient anomaly (10× attention grad) interact with the shift?
    - a: Unknown. Layer 0 is where one-hop induction is most natural, which is an argument for keeping it. Log layer-0 attention grad norm (baseline 0.2406 at step 100) and see whether it moves.
    - decides: shift in layer 0 vs start shifting at layer 1

known_tagged:
  -
    - text: Heads 3–5 means `k[:, 3:6]` in the `(B, H, T, 96)` key tensor after `k = torch.cat([self.k_norm(k), kr.expand(...)], dim=-1)`; values `v` are not touched.
    - prov: script
  -
    - text: `kr` comes from one rotary projection shared by all heads (`self.rotary` → `rd=32`, expanded to `n_head`), so shifting it for heads 3–5 only must act on the expanded tensor.
    - prov: script
  -
    - text: `k_norm` is per-token, so shifting before or after it gives the same result, except for the padded position.
    - prov: script
  -
    - text: `torch.roll` along T leaks token T−1 into position 0's key; pad with zeros instead.
    - prov: belief
  -
    - text: SDPA scale is 1/√96 (q last dim = 64 + 32); the content-logit ceiling at norm weight 1 is about 6.5.
    - prov: script
  -
    - text: Total steps at 50M tokens = 50e6 // 32768 = 1525; the predicted induction window (steps 150–300) is 4.9–9.8M tokens.
    - prov: script
  -
    - text: A 45-step smoke run has its own schedule: warmup is 1 step and cooldown starts at step 31. Baseline and candidate must use the same `--smoke-steps`.
    - prov: script
  -
    - text: Recall tokens are 13% of targets with loss 4.475 vs 6.704 at step 100; late-minus-early position loss is +0.459.
    - prov: measured
  -
    - text: Max attention logit at step 100 is 7.81 (layer 0) to 22.75 (layer 8); first-token attention mass is 0.0023–0.0059.
    - prov: measured
  -
    - text: The leak test should check that logits at positions ≤ i are bit-identical (in fp32) when tokens > i are randomised; include i = 0 and i = T−1.
    - prov: belief
  -
    - text: H3 put a shift SSM on keys specifically so one layer can do associative recall/induction (https://arxiv.org/abs/2212.14052).
    - prov: web
  -
    - text: Primer's depthwise convolution after the Q/K/V projections (a learned local shift) improved LM quality (https://arxiv.org/abs/2109.08668).
    - prov: web
  -
    - text: One-layer attention-only transformers cannot form induction heads without a previous-token mechanism; normally a previous-token head and an induction head span two layers (https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
    - prov: web
  -
    - text: No parameter change: non-embedding stays at 21.24M.
    - prov: script

open:
  -
    - q: Does the baseline form induction heads at all within 1525 steps, and at which step?
    - build: True
    - decides: trust the 45-step kill alone vs add a matched 300-step gate before seeds
  -
    - q: Is the cost on non-recall tokens from losing three standard heads per layer below 0.01 nats?
    - build: True
    - decides: 3 shifted heads per layer vs 1 per layer
  -
    - q: Which layers' shifted heads actually do induction (the L1 share per layer)?
    - build: True
    - decides: all 12 layers vs early layers only
  -
    - q: Does the zero-padded key at position 0 become an attention sink for heads 3–5?
    - build: True
    - decides: zero pad vs a learned null key
  -
    - q: How does `nano/exam.py` rebuild `Config` from `model.pt`?
    - build: True
    - decides: hardcoded shift vs a new config field
  -
    - q: What is the seed-to-seed std of the exam val loss? Is a 0.02 gain detectable at p < 0.01 with 3 seeds?
    - build: False
    - decides: 3 seeds vs 5 seeds
  -
    - q: Does the layer-0 attention gradient anomaly change when layer 0's heads 3–5 are shifted?
    - build: False
    - decides: include layer 0 vs start at layer 1

### New ideas born here
  - sc21d6f35ef
  - s8302812320
  - s40dab6db58
  - s545c7c8c3a
  - s0ab97a5c0e
  - sf23c585ac6
