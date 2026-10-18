# Results: improving the strongest open architecture at small scale

Date of the runs: 2026-10-13 to 2026-10-15. Hardware: one laptop RTX 4050 (6 GB). Every model trained here from scratch.

## In one paragraph

The HTSA method generated, filtered and tested ideas for improving a language-model architecture. Applied to the dense
skeleton of the strongest 2026 open models (DeepSeek-V4 and Kimi K3) scaled down to 21 million parameters, its best
combination lowered the held-out loss from **4.0939 to 4.0212**: **1.8 percent lower loss, about 7.0 percent lower
perplexity** (hybrid version; the first final version reached 4.0281, 1.6 percent / 6.4 percent), at the same parameter count, the same 50 million training tokens and the same data. One seed; the gap is
about twenty times the seed-to-seed spread measured on the plain recipe. On a plain GPT-2-style modern recipe, an earlier
combination gave a gain of 0.0089 loss (about 0.9 percent lower perplexity), verified over four seeds.

## How every model was trained and judged

- Data: FineWeb-Edu, GPT-2 tokenizer. Training reads 50 million tokens (1,525 optimizer steps) from the training shards.
- Exam: the harness scores the saved weights on 2 million held-out validation tokens that no training script ever reads.
- Every run goes through a sandbox launcher (no network, no subprocesses, no access to validation data); no run recorded
  a violation.
- Optimizer, learning-rate schedule, data order and the loss function are identical across all rows; only the
  architecture changes.

## The four rows

| Model | Seeds | Exam loss | vs frontier skeleton | Step time |
|---|---|---|---|---|
| Plain GPT-2-style modern recipe (Muon, RoPE, SwiGLU) | 5 | 4.1296 | | 1.0x |
| Plain recipe + the method's first combination | 4 | 4.1206 | | 1.5x |
| Frontier skeleton (DeepSeek-V4 / Kimi K3 dense skeleton at 21M) | 1 | 4.0939 | | 2.9x |
| Frontier skeleton + the method's final combination | 1 | 4.0281 | -0.0658 (-1.6% loss, -6.4% perplexity) | 2.7x |
| **Same, hybrid: one stream kept as the raw embedding, delay line on the other three** | 1 | **4.0212** | **-0.0727 (-1.8% loss, -7.0% perplexity)** | 2.6x |

Other full runs on the skeleton, for the record:

| Variant | Exam loss |
|---|---|
| + the seven ideas generated for the plain recipe | 4.0941 (no change) |
| + the four "safe" new ideas (exact skeleton at step 0) | 4.1002 (worse) |
| + five ideas generated for the skeleton, bundled | 4.1124 (worse; one idea, stream-relative writes, broke it) |
| + delay line + narrow keys | 4.0902 (slightly better) |
| + delay line with signed reads + value residual + narrow keys + soft-cap (final) | 4.0281 |
| + the same with an embedding stream and a three-stream delay line (final hybrid) | **4.0212** |

## What is in the final model

| Part | What it does | Quick-test gain (300 steps) | Prior art |
|---|---|---|---|
| Depth delay line | the four residual streams start as a cyclic shift, so stream j holds the state j layers back; a layer can read what each recent layer added | -0.065 | variant: DenseFormer (2024) gets a similar window with extra stored states; here it comes free from streams the model already has |
| Signed reads | layers combine the streams with free weights, so they can subtract one stored state from another | further -0.016 | variant: Hyper-Connections (2024) allow unconstrained weights; the aim of reading differences of lagged states is the change |
| Value residual | every layer mixes its attention values with the first layer's | -0.050 | known (ResFormer, modded-nanogpt speedrun) |
| Narrow keys, wide values | the compressed attention memory gives keys 32 and values 96 dimensions per head instead of 64 and 64 | -0.022 | variant: MLA already allows unequal sizes; "Thin Keys, Full Values" (2026) shrinks keys in standard attention |
| Logit soft-cap | squashes extreme prediction scores, 30 tanh(x / 30) | -0.016 | known (Gemma 2) |

Same parameter count as the skeleton (21.24M), no extra memory.

## How the ideas were found and filtered

1. The idea run read the skeleton's own code and its measured behaviour (three instrumented 100-step runs giving an
   anomaly map), drew a problem map of its parts, and ran 24 idea contexts: 275 sketches, 98 clusters, 10 questioned
   three levels deep with pre-mortems, 19 chosen for building. Idea generation on Claude Fable, questioning on Opus,
   research and clustering on Sonnet.
2. Unpicked sketches were also reviewed; the delay line came from there.
3. Each idea was built as its own file and trained 300 steps on the same seed and data order (18 quick tests).
   Clear winners were combined; anything neutral, worse, bigger or slower was left out.
4. A web search for prior art judged each winner by mechanism, not name.

### The 18 quick tests, gap to the skeleton at 300 steps (lower is better)

| Idea | Gap | | Idea | Gap |
|---|---|---|---|---|
| Delay line + signed reads | -0.081 | | Final stream read | +0.002 |
| Depth buckets with an embedding stream | -0.067 | | Decay-proof temperature | +0.004 |
| Delay line | -0.065 | | Gated key-value latent | +0.007 |
| Value residual | -0.050 | | Stronger stream mixing | +0.008 |
| Narrow keys, wide values | -0.022 | | Attention sink | +0.017 |
| Logit soft-cap | -0.016 | | Word-frequency prior | +0.021 |
| Hard stream reads | -0.007 | | One stream + 13th layer (1.7x faster) | +0.034 |
| Zero-initialised outputs | -0.006 | | Stream-relative MLP writes | +0.053 |
| Query against the latent | -0.003 | | | |
| Key salience | -0.001 | | | |

## What it says about the architecture

Every clear winner restructures the four parallel residual streams of the skeleton (delay line, signed reads, depth
buckets) or the flow of information between layers (value residual). Ideas that changed the inside of a single layer were
mostly neutral. The skeleton's four streams were nearly unused at initialisation; giving them a job is where the room was.
Known speedrun tricks mostly did not transfer to this design: of four tested, only the value residual helped.

## Honest limits

- One seed for every skeleton row. A second seed of the final model (about 75 minutes) is the cheapest way to confirm it.
- One model size, 21 million parameters, and 50 million tokens. Gains at this scale do not automatically hold at scale.
- The skeleton rows cost 2.7 to 2.9 times the plain recipe's compute per step; the comparison is at equal tokens, not
  equal compute.
- Novelty: of the ideas with a measured gain, one is new as far as the prior-art search found (the gain-tied lookup,
  plain-recipe run), four are variants of published work, two are known. Searches covered abstracts and summaries, not
  full papers.
- The skeleton is a dense approximation: no mixture of experts, no long-context machinery, latent attention with a
  128-wide latent, a static four-stream residual mixing in place of DeepSeek's dynamic one.

## Files

- Final model: `llmcandidates/final_model.py`; skeleton: `llmcandidates/strongest_scaled.py` (baseline v2 in
  `nano/train.py`); plain recipe archived as `nano/baselines/train_v1.py`.
- Runs: `D:\AGI-lab\runs\hand\ref\` (full seeds with weights and exams), `D:\AGI-lab\runs\hand\quick*` (quick tests).
- Idea run: `D:\AGI-lab\runs\skel-1__nano__gpt2_50m__scaffold__r0\` (sketches, chains, prior-art checks).
- Best model: `llmcandidates/final_hybrid.py` (one stream kept as the raw embedding, delay line on the other three), exam 4.0212.
