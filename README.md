# Epoch architecture run: improving the strongest open language-model skeleton on a laptop

The complete record of one research run by [Epoch](https://github.com/LuDeVing/epoch), a team of AI researcher
agents, on this task: take the dense skeleton of the strongest open language models of 2026 (DeepSeek-V4 and
Kimi K3), scaled down to 21 million parameters, and change its architecture so it learns better from the same data
and the same training budget. Everything here was produced between 2026-10-13 and 2026-10-15 on one laptop GPU
(RTX 4050, 6 GB).

## The result

Every model trained from scratch on 50 million tokens of FineWeb-Edu with the GPT-2 tokenizer, same optimizer,
schedule, data order and loss; graded on 2 million held-out tokens no training script reads. Only the architecture
differs. Lower loss is better.

| model | held-out loss | bits per byte | change |
|---|---|---|---|
| frontier skeleton (DeepSeek-V4 / Kimi K3 dense skeleton at 21M) | 4.0939 | 1.4447 | baseline |
| skeleton + the team's combination (`candidates/final_model.py`) | 4.0281 | 1.4215 | -1.6 percent loss, -6.4 percent perplexity |
| the same, hybrid (`candidates/final_hybrid.py`) | **4.0212** | **1.4190** | **-1.8 percent loss, -7.0 percent perplexity** |

Same parameter count (21.24 million), no extra memory, 2.6x the plain recipe's step time against the skeleton's
2.9x. One seed per row; the gap is about twenty times the seed-to-seed spread measured on the plain recipe. For
scale: in the NanoGPT speedrun the median record is a gain of about one percent, and a published architecture
paper such as value residual reports the same loss with about 16 percent fewer parameters.

The runs that did not make it, for the record:

| variant | held-out loss |
|---|---|
| skeleton + the seven ideas generated for the plain recipe | 4.0941 (no change) |
| skeleton + four "safe" new ideas, exact skeleton at step 0 | 4.1002 (worse) |
| skeleton + five ideas bundled, one of which broke it | 4.1124 (worse) |
| skeleton + delay line + narrow keys | 4.0902 (slightly better) |

## What is in the final model

| part | what it does | quick-test gain | prior art |
|---|---|---|---|
| depth delay line | the four residual streams start as a cyclic shift, so stream j holds the state j layers back; a layer can read what each recent layer added | -0.065 | variant of DenseFormer (2024) |
| signed reads | layers combine the streams with free weights, so they can subtract one stored state from another | further -0.016 | variant of Hyper-Connections (2024) |
| value residual | every layer mixes its attention values with the first layer's | -0.050 | known (ResFormer, modded-nanogpt) |
| narrow keys, wide values | 32 key and 96 value dimensions per head instead of 64 and 64 | -0.022 | variant of latent attention; "Thin Keys, Full Values" (2026) |
| logit soft-cap | 30 tanh(x / 30) on the output logits | -0.016 | known (Gemma 2) |

Every clear winner restructures the skeleton's four parallel residual streams or the flow between layers. Ideas that
changed the inside of a single layer were mostly neutral. The skeleton's four streams were nearly unused at
initialisation; giving them a job is where the room was.

## How the ideas were found

1. The team read the skeleton's code and three instrumented 100-step runs of its behaviour, drew a map of its parts,
   and ran 24 idea contexts: **275 sketches, 139 clusters, 10 questioned three levels deep with pre-mortems, 19
   chosen for building**. Idea generation on Claude Fable, questioning on Opus, research and clustering on Sonnet.
2. The unpicked sketches were reviewed too; the delay line, the largest single gain, came from there.
3. People, with a coding assistant, wrote each idea as its own file and trained it 300 steps on the same seed and
   data: 18 quick tests. Clear winners were combined; anything neutral, worse, bigger or slower was left out.
4. A web search judged each winner by mechanism, not name.

The 18 quick tests, loss over the last 50 of 300 steps (the skeleton: 5.1507; lower is better):

| idea | loss | | idea | loss |
|---|---|---|---|---|
| delay line + signed reads | 5.0702 | | hard stream reads | 5.1441 |
| depth buckets with an embedding stream | 5.0838 | | zero-initialised outputs | 5.1443 |
| delay line | 5.0860 | | query against the latent | 5.1482 |
| value residual | 5.1011 | | final stream read | 5.1525 |
| narrow keys, wide values | 5.1283 | | gated key-value latent | 5.1577 |
| logit soft-cap | 5.1351 | | stronger stream mixing | 5.1588 |
| attention sink | 5.1679 | | word-frequency prior | 5.1716 |
| one stream + 13th layer | 5.1844 | | stream-relative MLP writes | 5.2033 |

## What is in this repository

- `RESULTS.md`: the full write-up of the result, the rows, the limits.
- `idea_run/`: what the agents did. `SKETCHES.md` is every sketch by part and kind; `BY_HAND.md` the build briefs for
  the 19 chosen clusters; `problem_tree.json` the map; `ideas.jsonl` and `idea_clusters.json` the ideas and their
  groups; `deepen.json` the critics' chains; `selection.jsonl` the picks; `research/` the prior-art reading per part;
  `prompts/` and `calls/` every prompt sent and every reply received; `train_baseline.py` the recipe the run read.
- `candidates/`: the baseline (`strongest_scaled.py`, the skeleton), every single-idea file that was quick-tested,
  the combinations, and the two final models (`final_model.py`, `final_hybrid.py`). `NOTES.md` is the build log.
- `quick_tests/`: the four batches of 300-step quick tests with their results and logs.
- `reference_runs/`: the full 50-million-token runs of every row above: the exam, the metrics over training, the
  sandbox audit and the console output. The weights (162 MB each) are not included.

## Reproduce

The training and exam scripts live in the Epoch repository: `nano/train.py` (the skeleton is its baseline v2) and
`nano/exam.py`. With a CUDA build of torch and the FineWeb-Edu shards prepared by `nano/prepare_data.py`:

```
python -m nano.train --candidate candidates/final_hybrid.py --out runs/final_hybrid/seed0
python -m nano.exam  --run runs/final_hybrid/seed0
```

A full run takes about 75 minutes on an RTX 4050; a quick test about 15 minutes.

## Limits

- One seed for every skeleton row. A second seed of the final model is the cheapest way to confirm it.
- One model size and one token budget. Gains at 21 million parameters and 50 million tokens do not automatically
  hold at scale.
- The skeleton rows cost 2.6 to 2.9 times the plain recipe's compute per step; the comparison is at equal tokens.
- Of the ideas with a measured gain, four are the team's own variants of published mechanisms and two are known
  techniques it arrived at from the measurements; the prior-art search covered abstracts and summaries, not full
  papers.
- The skeleton is a dense approximation: no mixture of experts, no long-context machinery, a 128-wide latent, a
  static four-stream residual mixing in place of DeepSeek's dynamic one.
