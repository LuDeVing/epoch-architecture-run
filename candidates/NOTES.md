# Sequential build and check log

All 14 candidates are independent complete copies of train_baseline.py. Only architecture changes are allowed. No data was read, no training or optimizer steps were run, and tokens remains 50e6. Parameter cap: 23,367,590.4 (integer maximum 23,367,590). Checks use default model sizes, random tokens and a forward/backward pass, never smoke training. C4 full-step time/VRAM, 45-step predictions and C7 improvement are **not measured**.

Source discrepancies: BY_HAND.md has a global-reference sketch under the companded heading, omits ideas 3-6, and conflates shrink-std and semi-orthogonal initialization. Matching original records in ideas.jsonl/SKETCHES.md supply those requested mechanisms. All such choices are named in FIDELITY. Top-k follows the explicit hard-mask resolution in the interrogation.

The required IDEA comment for key=value contains the substring `val`; the grep audit records that mandatory metadata-only match separately. No new match is allowed in executable code or other comments.

## 1. Exact-transpose tying through the final gain -> transpose_tying.py

PLAN 1: GPT.forward multiplies the token lookup by GPT.norm.weight. Parameters: 0; nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: unchanged. No training timing is run.
PLAN 3: Pre-mortem handling: No pre-mortem objections are listed for this sketch; preserve the head, tied storage, initialization and optimizer.

Source: BY_HAND.md:5-15. FIDELITY: exact  The learned final gain is reused without a new parameter.

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 2 changed hunks including the required header. Full diff: `.checks/transpose_tying.diff`.
- `python transpose_tying.py --build-only`: {'status': 'PASS', 'params_total': 40560000, 'params_nonembed': 21243264, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.844311714; gradient norm 29.056274729 -> 29.056350781; max common-parameter norm difference 0.031666588; added gradients {}. Reach: PASS.
- Mechanism diagnostics: {'shared_initial_weights': 'bit-identical to baseline', 'gain_shared': True}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 1.7881393432617188e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/transpose_tying.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 1: PASS

## 2. Companded normalization with learned exponent -> companded_norm.py

PLAN 1: RMSNorm adds one alpha scalar per site and computes x/rms**alpha with the existing gain. Parameters: +25; nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: +1% predicted; manual scalar operations may add launch overhead. No training timing is run.
PLAN 3: Pre-mortem handling: No exponent pre-mortem is listed in BY_HAND; use alpha=1 at every site, FP32 log-RMS with epsilon, and preserve gain and all optimization. Do not combine the unrelated EMA reference sketch.

Source: ideas.jsonl:s637ea27dc3; SKETCHES.md:190; BY_HAND.md:431-437. FIDELITY: substituted  BY_HAND heading contains Global reference RMS; implement the requested exponent sketch s637ea27dc3 from ideas.jsonl instead.

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 2 changed hunks including the required header. Full diff: `.checks/companded_norm.diff`.
- `python companded_norm.py --build-only`: {'status': 'PASS', 'params_total': 40560025, 'params_nonembed': 21243289, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.844311714; gradient norm 29.056274729 -> 29.059435795; max common-parameter norm difference 0; added gradients {'blocks.0.n1.alpha': 0.27663540840148926, 'blocks.0.n2.alpha': 0.08396792411804199, 'blocks.1.n1.alpha': 0.09925752878189087, 'blocks.1.n2.alpha': 0.07460761070251465, 'blocks.2.n1.alpha': 0.0009193271398544312, 'blocks.2.n2.alpha': 0.07423822581768036, 'blocks.3.n1.alpha': 0.06597971171140671, 'blocks.3.n2.alpha': 0.010801410302519798, 'blocks.4.n1.alpha': 0.06928393989801407, 'blocks.4.n2.alpha': 0.010271910578012466, 'blocks.5.n1.alpha': 0.006407707929611206, 'blocks.5.n2.alpha': 0.010899040848016739, 'blocks.6.n1.alpha': 0.07600921392440796, 'blocks.6.n2.alpha': 0.03674602508544922, 'blocks.7.n1.alpha': 0.05886859819293022, 'blocks.7.n2.alpha': 0.03648165613412857, 'blocks.8.n1.alpha': 0.014043457806110382, 'blocks.8.n2.alpha': 0.02036970481276512, 'blocks.9.n1.alpha': 0.03599639981985092, 'blocks.9.n2.alpha': 0.02956382930278778, 'blocks.10.n1.alpha': 0.019382361322641373, 'blocks.10.n2.alpha': 0.014480473473668098, 'blocks.11.n1.alpha': 0.01418003998696804, 'blocks.11.n2.alpha': 0.024320993572473526, 'norm.alpha': 0.23147305846214294}. Reach: PASS.
- Mechanism diagnostics: {'shared_initial_weights': 'bit-identical to baseline', 'all_25_alpha_gradients_nonzero': True}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 1.7881393432617188e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/companded_norm.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 2: PASS

## 3. Magnitude-as-temperature final norm -> magnitude_temperature.py

PLAN 1: RMSNorm optionally adds a,b scalars; GPT.norm alone enables softplus(a*log_rms+b). Parameters: +2; nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: within noise predicted. No training timing is run.
PLAN 3: Pre-mortem handling: No pre-mortem is listed in BY_HAND; initialize a=0 and b=inverse_softplus(1), keep only the final norm modified, and compute epsilon-stabilized log-RMS in FP32. Confidence and late gains remain untested.

Source: ideas.jsonl:s3eed69d8e5; SKETCHES.md:185. FIDELITY: substituted  Sketch is absent from BY_HAND; use the matching full sketch s3eed69d8e5 in ideas.jsonl.

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 3 changed hunks including the required header. Full diff: `.checks/magnitude_temperature.diff`.
- `python magnitude_temperature.py --build-only`: {'status': 'PASS', 'params_total': 40560002, 'params_nonembed': 21243266, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.844311714; gradient norm 29.056274729 -> 29.056704995; max common-parameter norm difference 0; added gradients {'norm.a': 0.14631889760494232, 'norm.b': 0.059956952929496765}. Reach: PASS.
- Mechanism diagnostics: {'shared_initial_weights': 'bit-identical to baseline', 'temperature_starts_at_one_and_both_scalars_reached': True}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 1.7881393432617188e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/magnitude_temperature.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 3: PASS

## 4. Causal per-channel running scale before RMSNorm -> causal_channel_scale.py

PLAN 1: RMSNorm optionally applies detached per-channel prefix RMS; Block enables it on n1/n2, GPT.norm stays standard. Parameters: 0; nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: +5 to +10% predicted; prefix cumsum at 24 sites. No training timing is run.
PLAN 3: Pre-mortem handling: No pre-mortem is listed in BY_HAND; use detached FP32 prefix statistics per sequence, counts starting at one and epsilon, with no stored cross-batch state. Keep the final norm and per-channel gains unchanged.

Source: ideas.jsonl:s484ebb6a5f; SKETCHES.md:187. FIDELITY: substituted  Sketch is absent from BY_HAND; use matching sketch s484ebb6a5f in ideas.jsonl, including the current token in the causal prefix.

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 3 changed hunks including the required header. Full diff: `.checks/causal_channel_scale.diff`.
- `python causal_channel_scale.py --build-only`: {'status': 'PASS', 'params_total': 40560000, 'params_nonembed': 21243264, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.872729301; gradient norm 29.056274729 -> 46220814585.894187927; max common-parameter norm difference 3.25624955e+10; added gradients {}. Reach: PASS.
- Mechanism diagnostics: {'prebranch_sites': 24, 'detached_prefix_statistic': True}.
- Inference/causality/checkpoint checks: not completed.
- Failure: ('batch dependence', 0.09519076347351074).
- Detailed numerical evidence: `.checks/causal_channel_scale.json`. 45-step prediction not tested; C4 and C7 not certified.

IDEA 4 ATTEMPT 1: FAILED (resolved by the single repair below)

Idea 4 repair (one allowed fix): first attempt passed boundary/build/grep/reach, but extra batch-independence check found max logit delta 0.09519 and gradient norm 4.622e10. Increase only the detached channel statistic floor from 1e-6 to 1e-4; retain the ordinary RMSNorm epsilon and all other mechanisms. FIDELITY records this numeric stabilization. Rerun all checks below.

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 3 changed hunks including the required header. Full diff: `.checks/causal_channel_scale.diff`.
- `python causal_channel_scale.py --build-only`: {'status': 'PASS', 'params_total': 40560000, 'params_nonembed': 21243264, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.784531593; gradient norm 29.056274729 -> 2277.001199630; max common-parameter norm difference 1881.96627; added gradients {}. Reach: PASS.
- Mechanism diagnostics: {'prebranch_sites': 24, 'detached_prefix_statistic': True}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 6.079673767089844e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/causal_channel_scale.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 4: PASS

## 5. Attention starts as a similarity-weighted copy of the context -> copy_init.py

PLAN 1: GPT.__init__ initializes Wq/Wv semi-orthogonally, copies Wq to Wk and scaled Wv transpose to Wo after normal initialization. Parameters: 0; nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: unchanged during forward/backward; extra one-time QR initialization. No training timing is run.
PLAN 3: Pre-mortem handling: No pre-mortem is listed in BY_HAND; preserve Gaussian-matched Frobenius norms and residual depth scaling, initialize after all baseline draws, and copy weights without tying storage. Q/K and V/O shape compatibility is enforced for this initialization.

Source: ideas.jsonl:sea0c07b519; SKETCHES.md:255. FIDELITY: substituted  Sketch is absent from BY_HAND; use matching sketch sea0c07b519 in ideas.jsonl; copies are independent parameters after initialization.

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 2 changed hunks including the required header. Full diff: `.checks/copy_init.diff`.
- `python copy_init.py --build-only`: {'status': 'PASS', 'params_total': 40560000, 'params_nonembed': 21243264, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.920212746; gradient norm 29.056274729 -> 16.339906137; max common-parameter norm difference 4.6098547; added gradients {}. Reach: PASS.
- Mechanism diagnostics: {'matrices_checked': 48, 'max_normalized_Gram_error': 1.430511474609375e-06}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 1.3113021850585938e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/copy_init.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 5: PASS

## 6. Diagonal-free attention with a learned null key -> diagonal_free.py

PLAN 1: Attention adds a per-head null key, prepends a zero output entry and uses a strictly preceding-token causal mask. Parameters: +4,608; nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: +10 to +20% predicted; explicit mask can select a slower SDPA backend. No training timing is run.
PLAN 3: Pre-mortem handling: No pre-mortem is listed in BY_HAND; an always-open null column prevents all-masked rows at position zero, null output is exactly zero, self/future keys are excluded, and window/GQA behavior is retained.

Source: ideas.jsonl:s207d345789; SKETCHES.md:115. FIDELITY: substituted  Sketch is absent from BY_HAND; use matching sketch s207d345789 in ideas.jsonl; null key starts at zero and remains unrotated.

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 3 changed hunks including the required header. Full diff: `.checks/diagonal_free.diff`.
- `python diagonal_free.py --build-only`: {'status': 'PASS', 'params_total': 40564608, 'params_nonembed': 21247872, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.880378723; gradient norm 29.056274729 -> 28.330690539; max common-parameter norm difference 3.5472393; added gradients {'blocks.0.attn.null_k': 0.0037729954347014427, 'blocks.1.attn.null_k': 0.0027604836504906416, 'blocks.2.attn.null_k': 0.0023089684545993805, 'blocks.3.attn.null_k': 0.0018797388765960932, 'blocks.4.attn.null_k': 0.0015641427598893642, 'blocks.5.attn.null_k': 0.001478712190873921, 'blocks.6.attn.null_k': 0.0010224190773442388, 'blocks.7.attn.null_k': 0.0012994601856917143, 'blocks.8.attn.null_k': 0.001029128092341125, 'blocks.9.attn.null_k': 0.0008556327666155994, 'blocks.10.attn.null_k': 0.0010335645638406277, 'blocks.11.attn.null_k': 0.0008110790513455868}. Reach: PASS.
- Mechanism diagnostics: {'first_attention_output_exact_zero': True, 'null_key_gradients_nonzero': True}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 1.9073486328125e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/diagonal_free.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 6: PASS

## 7. Semi-orthogonal init of Muon matrices -> semi_orthogonal_init.py

PLAN 1: GPT.__init__ reinitializes all seven interior matrices per block semi-orthogonally after baseline initialization, retaining writer depth scaling. Parameters: 0; nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: unchanged during forward/backward; extra one-time QR initialization. No training timing is run.
PLAN 3: Pre-mortem handling: O1: Init erasure by Muon/decay is a scientific risk; do not add a second mechanism or claim an enduring gain without the prohibited training. O2: Near-null directions need not block gradients; preserve the optimizer and test only actual flat spectra, not that premise. O3: Embedding-dominated early loss remains a scientific risk; embeddings are bit-identical to baseline at this seed. O4: Cooldown and the seed-noise significance bar are untested; report build/reach success separately from C7. O5: Use gain=std*sqrt(max(rows,cols)), with writer std reduced by sqrt(2L); check every matrix entry RMS and Gram matrix. O6: Run orthogonal initialization only on block matrices after self.apply and writer initialization; never initialize the tied table this way. O7: Preserve all baseline embedding RNG draws by applying new initialization after all baseline initialization; do not claim paired training significance. O8: NS5 spectrum and initial entropy are unchanged optimizer behavior; lower initial entropy is not presumed. O9: Preserve this recipe scales/lr exactly; no transfer of speedrun outcome is asserted.

Source: BY_HAND.md:754-765; pre-mortem:2289-2360. FIDELITY: partial  Use BY_HAND general semi-orthogonal sketch (lines 754-765); the same title elsewhere describes a distinct shrink-std idea.

Objection mapping (same order as BY_HAND.md; supplied text sometimes ends mid-sentence):

- O1: Muon erases the init on its own schedule: each step adds a flat-spectrum update of spectral norm lr·scale ≈ 0.02, so by step 45 the cumulative update (~0.6) already exceeds the orthogonal init's single singular value (0.39), and Muon's decoupled decay (p *= 1 − 0.002 per step at peak lr) shrinks wha
- O2: The "blocked input directions" premise does not block learning: dW = δ·xᵀ does not depend on W's own spectrum, and Muon orthogonalizes it, so the first non-warmup updates fill the Gaussian's near-null directions (min sv ~0.001 → ≥0.02 after one full-lr step, cond ~40 after step 1, ~3 by step 30). Th
- O3: The early loss the idea targets is set by embeddings, not by Muon matrices: at step 100 rank-0-100 tokens (42% of targets) sit at 3.06 nats while rare tokens sit at 9-12, i.e. the 6.6→6.2 drop over steps 50-100 is unigram/bigram learning carried by the tied embedding under AdamW, which the idea leav
- O4: The exam is after cooldown; the proposer's own L4 predicts −0.000 to −0.010, and with per-seed val std ~0.01 at 21M/50M tokens a one-sided Welch test at p<0.01 on 3 seeds needs a true gap of roughly 0.015+. Even a real early head start that fully persists would not clear C7.
- O5: Gain slip on rectangular matrices: orthogonal_ yields entry RMS 1/sqrt(max(rows,cols)); a builder who writes sqrt(rows) (or sqrt(cols)) makes w_gate/w_up (1024×384) or w_down (384×1024) off by sqrt(1024/384) ≈ 1.63x, changing residual-branch scale and the 1/sqrt(2L) design. The planned sanity check 
- O6: Ordering/tying slip: putting orthogonal_ inside _init (keyed on nn.Linear) also hits head.weight, which is tok.weight under tie_embeddings, so the embedding gets a semi-orthogonal 50304×384 init and the AdamW side changes too; conversely putting the orthogonal loop before self.apply lets normal_ ove
- O7: The 2-seed 45-step smoke is not a paired comparison: orthogonal_ consumes a different RNG stream, so every other torch init (embedding, remaining matrices) differs from the baseline at the same seed; a 0.01-0.03 difference at step 45 is within init-draw variance, and matched-seed "improvement" would
- O8: Muon's NS5 updates are not exactly flat (singular values spread ~0.7-1.2) and the attention logits at init under flat q/k spectra are smaller and more uniform than under Gaussian (top singular directions no longer dominate), nudging layer-0 entropy even closer to its already-flat 0.975; any effect i
- O9: Attribution to scale: the speedrun lineage's orthogonal init rides on zero-init output projections and lr ~0.05 with a different depth scaling; here wo/w_down are Gaussian-scaled and lr is 0.02 with 30% cooldown, so the reported lineage gain does not transfer and was marginal there as well.

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 2 changed hunks including the required header. Full diff: `.checks/semi_orthogonal_init.diff`.
- `python semi_orthogonal_init.py --build-only`: {'status': 'PASS', 'params_total': 40560000, 'params_nonembed': 21243264, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.832565308; gradient norm 29.056274729 -> 30.723674884; max common-parameter norm difference 1.5252409; added gradients {}. Reach: PASS.
- Mechanism diagnostics: {'matrices_checked': 84, 'max_normalized_Gram_error': 1.430511474609375e-06}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 1.6689300537109375e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/semi_orthogonal_init.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 7: PASS

## 8. Low-rank bigram injection at head -> bigram_head.py

PLAN 1: GPT.__init__/forward add zero tok2 rows and a unit-gain bigram_proj after the final norm and before the tied head. Parameters: +1,622,016; nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: under 1% additional trunk work predicted. No training timing is run.
PLAN 3: Pre-mortem handling: O1: Redundancy with existing direct paths is an unresolved research risk; add precisely the independent token table without extra trunk changes. O2: Early-only gain is untested; keep the term permanent and make no full-budget claim. O3: Name table tok2 so the frozen filter routes it to AdamW; assert the actual optimizer groups. O4: Zero tok2 and initialize projection at std=1/sqrt(32), as interrogation prescribes, rather than the slow std=0.02 alternative. O5: Create extra modules after all baseline initialization so all common weights remain bit-identical at the same seed. O6: No C7 or seed-noise claim without training; record only build/reach checks. O7: Keep prescribed decay and zero rare rows; rare-row progress requires training and is untested. O8: Fix rank at 32 and verify the full table/projection count against the 10% cap. O9: Keep the after-norm addition faithfully; zero initialization prevents an initial temperature jump, late competition remains untested. O10: Make module unconditional in GPT so candidate Config round-trips cannot disable it; strictly rebuild/load in the audit.

Source: BY_HAND.md:1754-2131. FIDELITY: partial  Faithful rank-32 mechanism and initialization from BY_HAND interrogation; module is unconditional to avoid new Config fields outside the named boundary.

Objection mapping (same order as BY_HAND.md; supplied text sometimes ends mid-sentence):

- O1: The 1-token-to-head path already exists in the baseline: tok(idx) rides the residual stream untouched to the final norm, and block 0's MLP supplies the asymmetric low-rank map on it, so E2·P duplicates what E plus block 0 already compute; layer-0 grad norms will not move and the gap never opens. Che
- O2: The gain is confined to rank 0-1000 targets, which the trunk already drives to 3.06 nats by step 100 of 1525; the table front-loads statistics the trunk learns by step 300 anyway, so the matched-step gap shrinks to zero during the 70 percent of training that remains and the cooldown erases the rest.
- O3: E2 is a 2-D parameter whose name contains neither "tok" nor "head", so build_optimizers routes it to Muon: Newton-Schulz turns the sparse row gradient into a dense 50304-row update scaled by sqrt(50304/32) = 39.6 x lr, every token's row moves every step including unseen ones, and Muon's per-step dec
- O4: If E2 lands in AdamW instead, lr 3e-3 on zero-init rows means the term's RMS ratio to norm(x) stays under 0.02 at step 45 (rows need norm about 5 against P std 0.02 to reach 0.1), so the 45-step kill test fires for optimizer-speed reasons unrelated to the mechanism, and a passed or failed kill says 
- O5: Registering nn.Embedding(50304, 32) before the blocks consumes 1.6M torch RNG draws in self.apply, shifting every block's init; the 45-step comparison against 6.63 is then init-noise versus init-noise (about 0.03-0.05 nats at that stage), larger than the 0.02 kill margin, so the kill test is decided
- O6: The predicted exam gain of 0.005-0.03 nats sits at the seed-to-seed spread of this model at 50M tokens (about 0.01), the baseline exam loss is "unmeasured in the notes", and C7 demands p < 0.01 over 3 seeds; even a real 0.01 effect fails the Welch test more often than not. Check: pooled baseline exa
- O7: Rows for rare current tokens (half the vocab is seen under about 100 times in 50M tokens) receive too few Adam updates to leave zero while being decayed every step, so the table only stores frequent-current-token bigrams, precisely the ones the trunk already handles; the rank 10000+ bucket at 11.77 
- O8: param_counts subtracts only tok.weight, so the 1.62M table counts as non-embedding; --build-only reports 22.86M (7.6 percent over), which passes, but any follow-up widening past 32 (rank 48 = 2.4M) breaks C3; a builder who widens after a weak kill-test result trips the constraint. Check: --build-onl
- O9: The unnormalized term is added after the final norm whose weight grows with the residual RMS (x5.1 trunk growth); late in training the final-norm weight and E2·P scale compete for the logit temperature on tied E, and with the term bypassing the norm, logit magnitude on frequent bigrams can run up du
- O10: If the builder gates the module behind a new Config field, the exam rebuilds GPT(cfg) from the saved config and loads state_dict; a default-False flag plus a model.pt saved with the table present fails strict loading, or the flag defaults True and the exam silently differs from the smoke runs. Check

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 3 changed hunks including the required header. Full diff: `.checks/bigram_head.diff`.
- `python bigram_head.py --build-only`: {'status': 'PASS', 'params_total': 42182016, 'params_nonembed': 22865280, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.844311714; gradient norm 29.056274729 -> 29.056329044; max common-parameter norm difference 0; added gradients {'tok2.weight': 0.056182052940130234, 'bigram_proj.weight': 0.0}. Reach: PASS.
- Mechanism diagnostics: {'shared_initial_weights': 'bit-identical to baseline', 'rank': 32, 'table_AdamW': True, 'projection_Muon': True, 'zero_term_but_nonzero_table_gradient': True}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 1.7881393432617188e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/bigram_head.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 8: PASS

## 9. Drop value projection (key=value) with more heads -> key_is_value.py

PLAN 1: Config.n_head/n_kv_head become 9; Attention removes wv and fixes head width at 64; GPT.forward uses that rotary width. Parameters: +884,736; nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: about 1.15s predicted; objections estimate 1.25 to 1.35x baseline. No training timing is run.
PLAN 3: Pre-mortem handling: O1: Match/carry interference is an unresolved risk inherent in the sketch; retain raw key as carry features rather than adding a projection. O2: Effect below seed noise is untested; make no C7 success claim. O3: Keep frozen Muon shape-aware scaling; record the changed matrix shape rather than compensate via optimization edits. O4: No idle sink is added; the lack of a separate zero carry direction is an unresolved consequence of key=carry. O5: 45-step blindness remains; the random-token audit only establishes computational reach. O6: Record the possible 1.25-1.35x cost; no full-step timing or 4050 certification without training. O7: Use explicit head width 64 in Attention and GPT rope_cache; verify 576-wide projections and the exact parameter count. O8: Set both allowed Config head counts to 9, preserve the GQA repeat path, and strictly rebuild/load from saved Config. O9: Effective routing capacity is a research uncertainty; do not add match/carry splits beyond what learned weights can represent. O10: Assign carry tensor before RoPE and rotate only the query/key tensors; assert no wv and preserve unrotated carry.

Source: BY_HAND.md:938-1356. FIDELITY: exact  Nine 64-wide query/key heads carry the unrotated key, with the original SDPA scaling and no added temperature.

Objection mapping (same order as BY_HAND.md; supplied text sometimes ends mid-sentence):

- O1: Value capacity is halved, not free: with v = k every head must spend the same 64 dims on matching and on carrying, so a head that matches on "previous token identity" for induction can only carry that identity, not the token that followed it; the copy circuit needs two different projections per head
- O2: Below the noise floor by construction: the prediction is 0.01 nats and the pass bar is 0.005 nats at one-sided Welch p < 0.01 over 3 seeds. Seed std on 2M exam tokens at 50M-token training is on the order of 0.01, so three seeds give a standard error near 0.006, and a true 0.01 effect reaches p < 0.
- O3: Muon's shape scaling silently raises the q/k learning rate: wq and wk become 576x384, so the update scale max(1, rows/cols)^0.5 = 1.22 instead of 1.0, and the orthogonalized update now fills 384 singular directions at unit size in a wider matrix. Keys therefore grow faster than the tuned baseline, a
- O4: No attention sink is possible: the baseline lets the first token have an attractive key and a near-zero value, so heads can "attend to nothing." With v = k, the sink key that queries lock onto is also what gets copied into the residual, so every idle head injects a constant vector per layer. The mod
- O5: The 45-step smoke cannot see the mechanism: at 45 steps (1.5M tokens) loss is 6.6 and dominated by unigram and bigram statistics learned by embeddings and MLPs, so key=value with 9 heads passes the 6.70 kill bar trivially and the test is uninformative about the attention-capacity loss that only show
- O6: Step time is understated: SDPA cost scales with n_head x T^2 x head_dim, so 9 heads of 64 is 1.5x the attention FLOPs, and at d=384 with T=1024 attention is roughly a third of the forward. Projections grow 1.125x. Expect 1.25x to 1.35x step time, not 1.15x. This does not break C4 but it shrinks any 
- O7: Head width wiring slip: Attention computes hd = cfg.d_model // cfg.n_head, which is 42 for n_head=9, and GPT.forward builds the RoPE cache with the same expression. Setting --n-head 9 alone silently trains 9 heads of 42 (378-wide), with 21 rotary pairs, and the count lands near 21.4M instead of 22,1
- O8: GQA branch and config round-trip slip: n_kv_head defaults to 6, so with n_head=9 the repeat_interleave factor is 9 // 6 = 1 and SDPA gets 9 query heads against 6 key heads. If the builder fixes it by hardcoding 9 inside Attention instead of through cfg, model.pt stores n_head=6 and the exam's GPT(cf
- O9: More heads do not buy more routing when match dims are shared: each of the 9 heads now matches with the fraction of its 64 dims it can afford to not use for carrying, so effective match width per head drops to roughly 32 to 40 while head count rises 1.5x. Net routing capacity is flat or lower, and t
- O10: Pre-RoPE key as value is a different quantity from the key that matched: the score uses the rotated key but the copy uses the unrotated one, so the "key = value" identity used to motivate the mechanism does not hold position-wise, and the gradient from the value path has no rotation while the score 

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 5 changed hunks including the required header. Full diff: `.checks/key_is_value.diff`.
- `python key_is_value.py --build-only`: {'status': 'PASS', 'params_total': 41444736, 'params_nonembed': 22128000, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches ['# IDEA: Drop value projection (key=value) with more heads'].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.963850975; gradient norm 29.056274729 -> 32.281667263; max common-parameter norm difference 3.07557807; added gradients {}. Reach: PASS.
- Mechanism diagnostics: {'heads': 9, 'head_width': 64, 'raw_key_carry': True, 'no_wv': True}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 1.7881393432617188e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/key_is_value.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 9: PASS

## 10. Causal context-averaged mean-square normalizer -> context_norm.py

PLAN 1: RMSNorm.forward replaces the divisor at all 25 sites with half own and half strictly preceding 16-token mean-square. Parameters: 0; nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: +2 to +4% predicted; unfused kernels may exceed this estimate. No training timing is run.
PLAN 3: Pre-mortem handling: O1: Small magnitude modulation being absorbed remains untested; keep the exact half-own blend and test its sqrt(2) bound. O2: Keep the full cross-token gradient prescribed; train-away of magnitude variation is an unresolved research risk. O3: Use the contextual final norm as explicitly required; its temperature risk is recorded without silently substituting a standard final norm. O4: Outlier contagion is intrinsic to arithmetic averaging; retain it and do not invent clamps or geometric averaging. O5: Potential q/k-logit amplification is an untested stability risk; the blended norm bound is checked but not a claim of stable training. O6: Use padded prefix sums indexed at t and t-16 with no roll; explicitly check prefix invariance and zero gradients into future tokens. O7: Use actual predecessor counts, own statistic at t=0, FP32 squares/cumsum and epsilon; test T=1 and short prefixes for finite outputs. O8: Record manual kernel overhead as unmeasured; no fused extension or optimizer/data changes. O9: Record the kill threshold as a prediction only; no 45-step or C7 claim without training.

Source: BY_HAND.md:1358-1752. FIDELITY: exact  Use full gradients, FP32 prefix sums, actual predecessor counts, and own statistic alone at position zero.

Objection mapping (same order as BY_HAND.md; supplied text sometimes ends mid-sentence):

- O1: The blend caps the "preserved magnitude" to a post-norm RMS in (0, 1.41): with r = own/window, output RMS is sqrt(2r/(r+1)), so a token 4x its neighbours comes out 1.37 and one at 1/4 comes out 0.63. Inside the blocks the next Linear absorbs a ±30% per-token scale jitter as easily as it ignores it, 
- O2: The coupling creates a gradient path from each token's loss into its 16 predecessors' residual magnitude, and the cheapest way for the network to remove the noise that path injects is to equalize token magnitudes; the pre-norm CV then decays over training and the norm collapses to the baseline with 
- O3: The final norm before the tied head is also contextual, so the logit temperature now jitters by position according to sqrt(2r/(r+1)); unlike inside a block, nothing downstream can undo a per-position temperature error, and cross entropy pays for it directly on every token. Check by making only the f
- O4: Attention sinks and massive-activation tokens appear well after step 100 (first-token mass is 0.005 now, but position-0 residual norms of 50 to 1000x are standard in trained GPT-2s); with a 16-token window, positions 1 to 15 after such a token get their divisor inflated by up to sqrt(0.5 + 0.5·R/16)
- O5: q and k now scale per token by up to 1.41 each, so attention logits scale by up to 2x; the baseline already reaches max logit 20 at step 100 and grows through training, so late layers cross the stated 40 threshold, entropy collapses harder, and the late-training advantage reverses.
- O6: Implementation slip: the moving average built from a cumsum via torch.roll wraps the last 16 cumulative sums to the front, so positions 0 to 15 see future magnitudes; the TEST only checks single-window batch independence, which this passes, so the acausal statistic survives to the exam. Use F.pad wi
- O7: Implementation slip: "preceding 16 positions" that excludes the current token gives position 0 an empty window; dividing by a count of zero or by 16 unconditionally yields inf/NaN at position 0 or a 1.37x boost on the first positions, and a bf16 cumsum (if the builder squares x in bf16 before cumsum
- O8: Replacing fused F.rms_norm at 25 sites with an unfused fp32 square, cumsum, shift, blend, rsqrt and multiply adds six or more small kernels per site forward and backward; on an RTX 4050 with a 4x1024 micro-batch these launch-bound ops cost more than the predicted 2 to 4% but stay well under the 2x l
- O9: The KILL rule and the 45-step smoke measure a regime dominated by unigram and bigram learning where no norm variant differs by more than noise; the ladder will pass the idea on learn_z without any evidence that L1 through L4 hold, and the only informative measurement is the full-budget exam over 3 s

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 2 changed hunks including the required header. Full diff: `.checks/context_norm.diff`.
- `python context_norm.py --build-only`: {'status': 'PASS', 'params_total': 40560000, 'params_nonembed': 21243264, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.832102776; gradient norm 29.056274729 -> 27.606135246; max common-parameter norm difference 0.699880838; added gradients {}. Reach: PASS.
- Mechanism diagnostics: {'strictly_preceding_16_reference': True, 'output_RMS_bound': True, 'no_future_gradient': True, 'predecessor_gradients_nonzero': True}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 1.430511474609375e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/context_norm.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 10: PASS

## 11. Second pass over block subset (looped depth reuse) -> looped_depth.py

PLAN 1: Block adds alternate n1/n2 only for the upper six blocks; GPT.blocks construction and forward route the second pass through them. Parameters: +4,608 (6 blocks x 2 gains x 384); nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: about 1.37s predicted; six extra forwards/backwards. No training timing is run.
PLAN 3: Pre-mortem handling: O1: Keep two-depth gradients summed into the shared matrices and independent pass gains; conflicting roles remain an unresolved research risk. O2: Seed-noise and C7 significance are untested; build success does not establish the predicted gain. O3: Use original writer scales faithfully; verify the second pass executes six additional blocks, while weak learned contribution remains untested. O4: Cooldown closing any early gap is untested; do not alter schedule or add parameters to counter it. O5: No 45-step run is permitted; list the kill threshold without treating a random reach pass as evidence of learning. O6: Loop is unconditional in candidate GPT(cfg), with added gains registered in Block; strictly reload the graph from saved Config. O7: Block.forward selects alternate norms directly, never stacks them on ordinary norms; hook each to verify one invocation. O8: Record +50% block compute and unmeasured full-step VRAM/time; random audit is not a 4096-token C4 certification. O9: Repeat only upper-half indices 6-11 for the default twelve-layer Config, not blocks 1-6. O10: Leave global clipping and all optimization unchanged; the two-depth clip interaction is a scientific risk, not repaired by a forbidden optimizer edit.

Source: BY_HAND.md:347-429,859-936. FIDELITY: partial  Exactly repeat blocks 6-11 with separate gains; actual added count is 4,608 rather than the sketch estimate about 9k; graph is unconditional.

Objection mapping (same order as BY_HAND.md; supplied text sometimes ends mid-sentence):

- O1: Shared blocks 6–11 get the sum of two depths' gradients, and Muon orthogonalizes that sum into one fixed-norm update per step, so each shared matrix takes one step toward a compromise between two roles; over ~1526 steps it fits neither role as well as a dedicated block does, and 384 per-pass gains c
- O2: The predicted 0.01–0.03 gain sits on top of seed-to-seed std around 0.01 at 50M tokens with a 2M-token exam, so the low end of the prediction cannot pass a one-sided Welch test at p<0.01 on 3 seeds even if real.
- O3: The second pass writes into a residual stream already at RMS ~9 with w_o/w_down sized by 0.02/sqrt(24) and gradient norms at the top that are already 14x smaller than at layer 0; the pass-2 contribution is a tiny perturbation of the stream, so the model behaves as 12 layers plus noise and the gap to
- O4: The gap is flat through the plateau (both models learn unigram/bigram statistics via embeddings, which are unchanged) and only could open late, but during the 30% cooldown the dedicated baseline blocks converge while the shared blocks are still averaging two gradient fields, so any early advantage c
- O5: The 45-step smoke kill rule (loss ≤ 6.66, step ≤ 1.9s) is blind: the idea itself predicts smoke loss equal to baseline, so the ladder's early gate cannot discriminate and the first real signal is the full run.
- O6: Exam fidelity slip: if the loop is switched on by anything not in the Config dataclass (module constant, env var, an argparse flag added outside the asdict loop), the harness's GPT(cfg) builds a plain 12-block graph, strict load fails on the pass-2 norm keys or non-strict load silently drops them, a
- O7: Builder slip in the second pass itself: calling b(x, cos, sin) twice without routing the alternate n1/n2, or applying pass-2 gains as an extra RMSNorm stacked on n1 (double normalization), or forgetting that param_counts/Muon selection are fine but the saved state_dict key names for the new norms mu
- O8: Hidden compute cost: six extra block forwards and backwards add ~50% of block FLOPs and ~0.4–0.5 GB of saved activations at 4096 tokens per micro-batch; the baseline peak is not stated, and if it is near 5.1 GB the extra pass crosses 5.6 GB, while the step_ratio lands near 1.4–1.5x for no metric ret
- O9: Variant note: the blocks 1–6 variant is worse than the chosen 6–11 one; those blocks have near-flat attention (entropy 0.975) tuned to a low-RMS stream and act as broad mixers, so re-applying them at the top smears the sharpened upper representation. The 6–11 variant is the one to keep, and it still
- O10: grad_clip=1.0 is already active in the baseline (layer-0 attention grad alone is 0.91); the summed pass-2 gradients raise the global norm and the clip scales all gradients down further; Muon and Adam are largely scale-invariant so this is a small effect, but it removes any hope that the second pass 

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 4 changed hunks including the required header. Full diff: `.checks/looped_depth.diff`.
- `python looped_depth.py --build-only`: {'status': 'PASS', 'params_total': 40564608, 'params_nonembed': 21247872, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.842951775; gradient norm 29.056274729 -> 31.631741728; max common-parameter norm difference 2.09565878; added gradients {'blocks.6.n1_second.weight': 0.012205163016915321, 'blocks.6.n2_second.weight': 0.006516776978969574, 'blocks.7.n1_second.weight': 0.011628544889390469, 'blocks.7.n2_second.weight': 0.006054328754544258, 'blocks.8.n1_second.weight': 0.012573610059916973, 'blocks.8.n2_second.weight': 0.006208168808370829, 'blocks.9.n1_second.weight': 0.011723577976226807, 'blocks.9.n2_second.weight': 0.006231887266039848, 'blocks.10.n1_second.weight': 0.011100239120423794, 'blocks.10.n2_second.weight': 0.005943601951003075, 'blocks.11.n1_second.weight': 0.011951460503041744, 'blocks.11.n2_second.weight': 0.005645188968628645}. Reach: PASS.
- Mechanism diagnostics: {'shared_initial_weights': 'bit-identical to baseline', 'block_applications': 18, 'separate_norm_sites_called_once': 36, 'repeated_indices': [6, 7, 8, 9, 10, 11]}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 1.6689300537109375e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/looped_depth.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 11: PASS

## 12. Biorthogonal input lens on tied matrix -> input_lens.py

PLAN 1: GPT.__init__ adds identity lens after baseline initialization; GPT.forward reads the lookup through it. Parameters: +147,456; nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: within noise predicted; one extra 384x384 projection. No training timing is run.
PLAN 3: Pre-mortem handling: O1: Redundancy in branch linears remains a scientific risk; place the lens on the raw residual input, where the direct path differs. O2: An early gap closing is untested; preserve the lens through inference and do not claim final improvement. O3: Keep ordinary Muon decay toward zero as explicitly prescribed; identity/spectral drift during training remains untested. O4: Create and identity-initialize lens after all baseline draws; name it lens so the frozen optimizer filter selects Muon. O5: Candidate GPT(Config) always contains lens and passes strict state reload; compatibility with an unavailable fixed harness module cannot be certified. O6: Gradient relocation is expected; confirm nonzero lens gradients without claiming anomaly removal or modifying clipping. O7: Verify +147,456 parameters and identity forward; large Muon updates and sustained scale growth require training and are untested.

Source: BY_HAND.md:17-255,2753-2985. FIDELITY: exact  nn.Linear represents L transposed in storage so lens(E[idx]) equals E[idx] @ L, with identity initialization and ordinary Muon routing.

Objection mapping (same order as BY_HAND.md; supplied text sometimes ends mid-sentence):

- O1: L is almost entirely redundant with block 0: block 0 reads n1(E L) through wq/wk/wv/w_gate/w_up, and since RMSNorm is a per-token scalar, wq∘norm∘L ≈ (wq L)∘norm, so the existing linears already span every reframing L can offer; the only non-redundant effect is on the raw embedding's share of the re
- O2: The bigram direct path E L Eᵀ matters only while the model is in the bigram regime (loss ~6 at step 100); by the end of the budget context modelling dominates, the blocks have learned the reframing anyway, and the early 0.01–0.02 gain closes. The chain itself names this as the failure pattern. Check
- O3: Muon weight decay is p.mul_(1 - lr·wd) toward zero, not toward identity. Summed lr multipliers over 1525 steps ≈1280, so the identity component of L decays to exp(-0.002·1280) ≈ 8% and L becomes a Muon-built matrix with singular values drifting toward ~1/wd ≈10 in consistent directions. The input is
- O4: Builder slip: if L is an nn.Linear, self.apply(self._init) runs after the identity init and overwrites L with N(0, 0.02), so the injected vector is a random projection with norm ≈0.15 and the direct path is scrambled from step 0; identity init only survives if written after apply. A second slip: any
- O5: The exam rebuilds GPT(cfg) and loads model.pt; if nano/exam.py imports GPT from the harness's fixed module rather than the candidate file, the extra 384x384 key is either rejected (strict load fails) or silently dropped, and the weights are scored with x = E[idx] while every block was trained on E L
- O6: Layer-0 attention gradient (0.9077) is anomalous because block 0 must reframe a self-predicting input; but L is inserted on exactly the path block 0 already linearly reads, so the anomaly moves into L's gradient rather than disappearing, and Muon's orthogonalization makes L's update size shape-fixed
- O7: Parameter and step cost are real but negligible (+147k, one 384x384 matmul), so the ladder's cheap rungs pass trivially and give no evidence either way; the cost that is not negligible is that a square L under Muon receives a full-rank unit-singular-value update every step, so with lr 0.02 it moves 

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 3 changed hunks including the required header. Full diff: `.checks/input_lens.diff`.
- `python input_lens.py --build-only`: {'status': 'PASS', 'params_total': 40707456, 'params_nonembed': 21390720, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.844311714; gradient norm 29.056274729 -> 29.083804747; max common-parameter norm difference 0; added gradients {'lens.weight': 1.2651472091674805}. Reach: PASS.
- Mechanism diagnostics: {'shared_initial_weights': 'bit-identical to baseline', 'identity_lens': True, 'lens_Muon': True, 'lens_gradient_nonzero': True}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 1.7881393432617188e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/input_lens.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 12: PASS

## 13. Hard top-k activation sparsity -> topk_mlp.py

PLAN 1: Config.d_ff becomes 1152; MLP.forward keeps the largest 288 gated activations in magnitude with a detached hard selection. Parameters: +1,769,472; nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: within noise predicted; objections estimate +15 to +30% from topk/scatter. No training timing is run.
PLAN 3: Pre-mortem handling: O1: Follow the interrogation hard-mask resolution: no straight-through surrogate, direct gradients only through selected activations. O2: Apply selection unconditionally in MLP.forward, equally in train and inference; test both paths. O3: Rich-get-richer/dead units are unresolved risks; do not add load balancing or decay changes outside this sketch. O4: Keep frozen Muon whole-matrix updates; winner-only direct gradients do not imply winner-only optimizer updates. O5: Post-hoc large-model parity does not certify sparsity from initialization; no learned-quality claim is made. O6: 45-step blindness remains; no training means no claim of late recovery or specialization. O7: Rare-token gains are untested; random-token reach proves only that the changed MLP computes loss gradients. O8: Record possible +15-30% kernel/activation cost; use dense topk/scatter as the requested mechanism, without external kernels. O9: Keep shape-aware scaling and clipping unchanged; the altered effective learning rate is an unresolved interaction.

Source: BY_HAND.md:2362-2751; mask resolution:2418-2419. FIDELITY: partial  BY_HAND sketch says straight-through but interrogation explicitly resolves it to a hard winner-only mask; implement that resolution.

Objection mapping (same order as BY_HAND.md; supplied text sometimes ends mid-sentence):

- O1: The idea contradicts itself: MECHANISM says "straight-through mask", L2 says "no straight-through". A builder who implements STE (mask applied to the forward value, gradient passed through to all 1152 units) gets a dense gradient, every unit trains on every token, and the whole specialization story 
- O2: The mask is built from `self.training` or a `topk` flag that the exam path never sets, so model.eval() in nano/exam.py runs a dense 1152-wide forward on weights that only ever saw 288 live units; the residual-stream scale and w_down columns are calibrated for the masked regime and the exam loss is f
- O3: No load balancing and weight decay 0.1 on Muon weights (lr 0.02, 1525 steps, ~e^-3 shrink for a never-winning unit) make rich-get-richer a ratchet: a unit that loses early gets smaller gate/up rows, produces smaller
- O4: Muon orthogonalizes the whole (1152x384) gradient matrix, so the update to a rarely winning unit's row is rescaled to the same singular magnitude as a frequently winning one; "winner-only gradient" does not become "winner-only update", and sparse-gradient rows get amplified noise instead of speciali
- O5: The Sparse-LLM parity result is about post-hoc activation sparsity on large, well-trained models, where 75-90% of the SwiGLU hidden carries negligible energy. At 21M params and 2.4 tokens per parameter this model is nowhere near that regime: every unit is still informative, so the top-288 cut remove
- O6: The 45-step and 100-step gates are at 3% and 6.5% of the 1525-step budget, when the loss is dominated by unigram/bigram statistics that the embedding (AdamW) learns and the MLP barely matters. Parity at step 45 proves nothing about the end, and the sparsity penalty is largest exactly when the model 
- O7: Rare-token loss at 11.77 (rank 10000+) at step 100 is essentially unigram loss and is driven by the tied embedding, not by which hidden unit fires; "specialized units serve rare contexts" has no mechanism to move those buckets by 0.05-0.2 nats in a model that sees each rare token a handful of times 
- O8: Per-token topk over the last dim (4096x1152, k=288) runs 12 times per micro-batch, 8 micro-batches per step, plus an int64 index tensor, a scatter and an extra saved mask per layer for backward; without torch.compile this is launch-bound and can add 15-30% to step time and a few hundred MB of peak m
- O9: Muon's shape-aware scale is sqrt(1152/384) instead of sqrt(1024/384), so gate/up rows get ~4% larger updates per step while only 25% of their outputs ever reach the loss; combined with grad_clip 1.0 renormalizing the smaller MLP gradients, the effective MLP learning rate shifts in an unplanned direc

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 3 changed hunks including the required header. Full diff: `.checks/topk_mlp.diff`.
- `python topk_mlp.py --build-only`: {'status': 'PASS', 'params_total': 42329472, 'params_nonembed': 23012736, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.896220207; gradient norm 29.056274729 -> 27.336660628; max common-parameter norm difference 1.2106905; added gradients {}. Reach: PASS.
- Mechanism diagnostics: {'hidden': 1152, 'selected': 288, 'actual_forward_nonzeros': 288, 'unselected_direct_gradients_zero': True}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 1.6689300537109375e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/topk_mlp.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 13: PASS

## 14. Document-relative rope positions -> doc_relative_rope.py

PLAN 1: GPT.forward derives sequence-specific positions from token 50256 and gathers caches; apply_rope accepts batch-specific caches. Parameters: 0; nothing removed unless stated in CHANGE.
PLAN 2: Expected step-time effect: under +3% predicted; objections allow +5 to +8% kernel overhead. No training timing is run.
PLAN 3: Pre-mortem handling: O1: Acknowledge that ordinary RoPE already preserves intra-document differences; explicitly test this and change only cross-document differences. O2: Negative offsets need not suppress attention; retain the proposed mechanism without claiming they are implausible. O3: Whether long-range distraction matters at this scale is untested; no research-success claim follows the audit. O4: Boundary dilution and seed-noise remain unresolved; no C7 statement without training. O5: Start-circuit reuse/early-only improvement is untested; report predicted gains only. O6: Offset-distribution changes are intentional; train and target-free inference use the identical reset calculation. O7: Gather (B,T,D/2) caches and unsqueeze to (B,1,T,D/2); retain standard 2-D handling and test distinct boundaries for B=2. O8: Record cummax/gather launch cost as unmeasured; no full-step timing claim.

Source: BY_HAND.md:257-334,785-857. FIDELITY: exact  Reset the token after EOT to position zero, retain EOT in the preceding document, and add no document attention mask.

Objection mapping (same order as BY_HAND.md; supplied text sometimes ends mid-sentence):

- O1: The mechanism's core claim is false for RoPE: q·k after rotation depends only on (pos_q − pos_k), so intra-document relative offsets are already identical across windows under absolute indexing. The reset changes nothing for in-document pairs; the only change is that cross-document keys get negative
- O2: Negative offsets are not "implausible": cos(−k)=cos(k) and the learned score on each rotary pair is close to symmetric early in training, so a previous-document key at offset −k looks like an in-document key at +k. The low-frequency pairs in particular barely rotate over 1024 positions, so the previ
- O3: The baseline's attention is nearly flat (entropy 0.83–0.975 across layers at step 100) and 50M tokens into a 12-layer d=384 model, long-range attention is barely used, so cross-document "distraction" costs the baseline very little loss. The idea fixes a problem the model at this scale does not have;
- O4: Any benefit shows up only on tokens after the first end-of-text in a window, about half of all positions, and the predicted intra-document gain of 0.003–0.01 is already diluted by half before the exam; the pooled seed-to-seed spread on this setup is on the order of 0.01 nats, so the Welch test at 3 
- O5: Early in training, positions after a reset look like the first positions of a window, which the baseline already predicts best (0–64: 5.941 vs late 6.405 at step 100), so a matched-step smoke advantage at 45–100 steps comes from the model reusing its "near the start" circuits, not from down-weightin
- O6: The reset shifts the training distribution of offsets: documents in FineWeb-Edu average well under 1024 tokens, so large positive offsets (500–1023) become rare and appear instead as negative offsets. The exam's 1024-window packing is identical, so this is not a train/test mismatch, but the high-fre
- O7: Builder slip: apply_rope slices the cache with cos[: x.size(2)] on dim 0; a per-sequence cache of shape (B,T,D/2) must be gathered, unsqueezed to (B,1,T,D/2), and not sliced on dim 0 (with B=4 the slice silently keeps 4 rows, then broadcasting against (B,6,T,32) errors or, at B=1, silently drops the
- O8: Hidden cost is small but not zero: cummax-based position recovery and a (B,T,D/2) gather every forward run in eager mode on every micro-batch (eight per step) and add CPU-side kernel launches to a step that is launch-bound at d=384; the 3 percent prediction can become 5–8 percent, still within C4 bu

Checks (no training):

- Diff: PASS: metadata + permitted Model classes/functions; only allowed Config fields differ; 3 changed hunks including the required header. Full diff: `.checks/doc_relative_rope.diff`.
- `python doc_relative_rope.py --build-only`: {'status': 'PASS', 'params_total': 40560000, 'params_nonembed': 21243264, 'cap': 23367590.400000002}.
- Grep for all nine requested strings: PASS: zero new matching source lines; baseline/source matches 17/17; metadata-only matches [].
- One random-token forward/backward each, baseline and candidate, on cuda, full default models at B=2,T=24: loss 10.844311714 -> 10.847058296; gradient norm 29.056274729 -> 29.067009815; max common-parameter norm difference 0.0276861191; added gradients {}. Reach: PASS.
- Mechanism diagnostics: {'positions': [[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], 'boundary_reset_after_EOT': True, 'intra_document_differences_unchanged': True, 'cross_document_differences_changed': True}.
- Inference/causality/checkpoint checks: {'status': 'PASS', 'causal_prefix_max_delta': 0.0, 'batch_max_delta': 1.5497207641601562e-06, 'T1_finite': True, 'strict_Config_state_reload': 'identical logits', 'train_eval_max_delta': 0.0}.
- Detailed numerical evidence: `.checks/doc_relative_rope.json`. 45-step prediction not tested; C4 and C7 not certified.

END IDEA 14: PASS

## Final results

All 14 final candidates passed the boundary, build/count, source grep and random-token reach checks. Idea 4 needed one repair; the initial failure and revised floor are recorded above. The full literal grep has one additional line for idea 9: its mandatory `IDEA` comment contains `value` and `key=value`. It has zero new matches in executable code or other comments; this metadata exception is not hidden. Diffs were produced with Python difflib (PowerShell `diff` is a Compare-Object alias); the nine-string grep uses the exact requested regex, with matching lines compared as multisets.

| # | File | Non-embedding params | Change | Boundary | Build/cap | Source grep | Reach | Final status |
|---|---|---:|---:|---|---|---|---|---|
| 1 | `transpose_tying.py` | 21,243,264 | +0.0000% | PASS | PASS | PASS | PASS | PASS |
| 2 | `companded_norm.py` | 21,243,289 | +0.0001% | PASS | PASS | PASS | PASS | PASS |
| 3 | `magnitude_temperature.py` | 21,243,266 | +0.0000% | PASS | PASS | PASS | PASS | PASS |
| 4 | `causal_channel_scale.py` | 21,243,264 | +0.0000% | PASS | PASS | PASS | PASS | PASS after one repair |
| 5 | `copy_init.py` | 21,243,264 | +0.0000% | PASS | PASS | PASS | PASS | PASS |
| 6 | `diagonal_free.py` | 21,247,872 | +0.0217% | PASS | PASS | PASS | PASS | PASS |
| 7 | `semi_orthogonal_init.py` | 21,243,264 | +0.0000% | PASS | PASS | PASS | PASS | PASS |
| 8 | `bigram_head.py` | 22,865,280 | +7.6354% | PASS | PASS | PASS | PASS | PASS |
| 9 | `key_is_value.py` | 22,128,000 | +4.1648% | PASS | PASS | PASS; header exception | PASS | PASS |
| 10 | `context_norm.py` | 21,243,264 | +0.0000% | PASS | PASS | PASS | PASS | PASS |
| 11 | `looped_depth.py` | 21,247,872 | +0.0217% | PASS | PASS | PASS | PASS | PASS |
| 12 | `input_lens.py` | 21,390,720 | +0.6941% | PASS | PASS | PASS | PASS | PASS |
| 13 | `topk_mlp.py` | 23,012,736 | +8.3296% | PASS | PASS | PASS | PASS | PASS |
| 14 | `doc_relative_rope.py` | 21,243,264 | +0.0000% | PASS | PASS | PASS | PASS | PASS |

The maximum count is 23,012,736 (+8.3296%), below the allowed 23,367,590 integer cap. All candidates retain the original training, optimizer, data, objective, evaluation, CLI, output and 50e6-token recipe code. Target-free inference, causal prefix invariance, batch independence, finite T=1 outputs and strict Config/state_dict reload were also checked. Reported gradient norms are FP32 random-token diagnostics, not trained-quality measurements. The repaired channel-scale model still has a large initial gradient norm (2277 versus baseline 29.06); passing the batch numeric check does not establish stable learning under the unchanged global clip.

No 45-step smoke, optimizer update, data read, training, full-step timing/VRAM measurement or exam was run. C4 and C7 remain unverified; every PREDICTION is a sketch expectation rather than an observed result. All diffs, literal matching-line dumps, build logs and detailed audit JSON are in `.checks/`; `.checks/build_ideas.py` contains the sequential preparation and verification logic.

# Two-agent candidates

Agents worked independently; their initial evidence is followed by the governing parent audit. All 14 singles and baseline hash-match the previous run.

## Agent A research

- DeepSeek V4 Pro: width 7168, 61 layers, 128 heads, head dimension 512, rotary 64, query rank 1536, output rank 1024; RMS epsilon 1e-6, theta 10000; 4 hyperconnection streams, 20 Sinkhorn iterations. Source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json
- V4 experts: 384 routed + 1 shared, 6 selected, expert width 3072, sparse index 64 heads x 128 and top-k 1024, context 1048576, one next-token extra layer. Same primary config source.
- Kimi K3: width 7168, 93 layers, 96 heads; content 128, rotary 64, latent 512, query rank 1536; 896 experts with 16 active and 2 shared; expert width 3072; 1M context. Hybrid KDA head size 128, full attention every fourth layer plus last, attention residual block size 12, RMS epsilon 1e-5, SiTU activation. Source: https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json
- Existence verified in primary publisher sources; strongest ranking is publisher claim, not established independently. V4 official release states 1.6T total / 49B active: https://deepseek.com/en/news/v4-preview/
- Conflict: explicit requested MLA is not V4 token-compressed attention; follow requested dense MLA with 128 latent and decoupled 32 rotary. Kimi hybrid attention and residuals excluded in favor of DeepSeek.
- Static learned 4-stream doubly-stochastic mixing approximates dynamic mHC; no dynamic map source is claimed. FFN 1077 selected arithmetically for matching baseline. Initializer 0.02 is sourced; scaled output initialization remains baseline rather than unsupported zero fallback.

## Agent A checks

- Frozen optimizer through EOF byte-identical; imports and original docstring byte-identical; changed model hunks architecture-only plus Config.d_ff and mandated header.
- Random default model, CPU float32 batch 1 x 16: [(10.981002807617188, 50.13225100405817, (40560000, 21243264)), (10.733691215515137, 51.6748020643841, (40849344, 21532608))]; finite; ratio 1.0307696349043067.
- Full default dimensions retained; short random sequence is for reach only, no optimizer steps.
- Smoke deferred to parent: configured real-data directory absent.

- Repair 1: initial FFN 1098 yielded 21,532,608 (+1.36%), failing 1% matching; reduced to 1077, resulting 21,242,304 (-0.00452%). Parent must rerun reach on final revision.

## Agent B choices

Included eight ideas: doc_relative_rope, diagonal_free, causal_channel_scale, looped_depth, input_lens, bigram_head, magnitude_temperature, transpose_tying.
- copy_init: conflicts with diagonal-free attention.
- semi_orthogonal_init: conflicts with chosen attention layout.
- key_is_value: bigram plus nine-head layout exceeds cap; weaker early prediction than bigram.
- topk_mlp: bigram plus wider sparse MLP exceeds cap; weakest early prediction.
- companded_norm: final exponent overlaps final temperature; channel scale leaves final path independent.
- context_norm: changes final divisor and overlaps final temperature.

The rank-32 bigram has the strongest numeric early prediction (6.5309�6.6109); top-k predicts a lag and key=value only within 0.03 of baseline. Both alternatives individually exceed the cap when paired with bigram. Channel scaling is confined to block norms, final temperature to GPT.norm; the lens follows the gain.

## Agent B checks

- Build PASS: total 42,338,690; non-embedding 23,021,954 <=23,367,590.
- Boundary PASS: raw-byte prefix excluding header through Config and suffix beginning param_counts identical.
- Grep: executable code identical counts; literal full-file grep has one added val substring in mandatory EXCLUDED filename key_is_value.py. Other eight counts unchanged.
- Reach PASS: full default architecture CPU float32, seed 0, same random 1x32 tokens, EOT at 16; baseline/combined (loss,norm) [(10.990711212158203, 36.89276885986328), (10.921869277954102, 347.77410888671875)]; norm ratio 9.426619894205674. All gradients finite.
- Merge repair: second_pass argument moved from Attention to Block after generic anchor matched wrong constructor.
- GPU smoke awaiting parent; real shard path absent.


Research qualification: Kimi moe_intermediate_size=3072 differs from routed_expert_hidden_size=3584. Primary publisher configs were also checked by the parent; strongest ranking is not independently established.

## Combined

Final seven: doc_relative_rope, diagonal_free, looped_depth, input_lens, bigram_head, magnitude_temperature, transpose_tying. Lens follows shared gain. Reference mechanisms preserved; other choices and cap exclusions above apply. The final temperature overlaps the final divisors in companded/context norms.

## Independent parent audit

- Combined parent repair 1: gradient ratio 12.7126196853 on CUDA B=2,T=24 exceeded 10x. Removing only causal channel scaling reduced it to 1.1128378691. Imported the temperature-only RMSNorm directly from its reference; seven ideas remain. Original source retained in .checks/combined_all.before_parent_repair.py.
- Scaled parent repair 1: identical initial streams with uniform writes/read weights leave mixing inert. Asymmetric write-logit rows [0.03,-0.01,-0.01,-0.01] and [-0.01,0.03,-0.01,-0.01] break this symmetry, providing useful stream-read gradients (maximum norm 0.0001473). Mixing gradients remain near zero initially while reads are uniform, but can develop after reads learn. This local initialization is named in FIDELITY. Parameter count unchanged. Agent FFN-count repair is recorded separately above.
- Parent reran --build-only, unified diff/byte-boundary, nine-string grep, and one random forward/backward each for baseline and candidate on both final files. Default model dimensions, CUDA FP32, B=2,T=24, distinct EOT positions. Frozen prefix excluding permitted Config lines and mandatory header, plus param_counts through EOF, are byte-identical. Evidence: .checks/*.parent.diff, *.parent.build.txt and *.parent.json.
- strongest_scaled.py: baseline/candidate loss 10.837491035/10.959022522; finite gradient norms 29.149704939/28.721045750; ratio 0.985294562; causal-prefix max delta 0.0.
- combined_all.py: baseline/candidate loss 10.837491035/10.862225533; finite gradient norms 29.149704939/32.438895530; ratio 1.112837869; causal-prefix max delta 0.0.
- Full literal grep: scaled file has no new lines; combined file adds only mandatory EXCLUDED filename key_is_value.py, which contains val. No executable or other-comment matches added.
- Ran python strongest_scaled.py --smoke-steps 3 and python combined_all.py --smoke-steps 3 on GPU with separate out directories. Both failed before the training loop: FileNotFoundError, no train shards in D:\AGI-lab\data\fineweb. Step time and peak VRAM are unmeasured; the 2.14-second limit is UNVERIFIED. No synthetic data, changed Shards, or substitute dataset. Zero optimizer steps ran. Token budget stays 50e6. This cannot be repaired within the frozen-data rules without the real shards.

| File | Non-embedding params | Boundary | Build/cap | Grep | Reach / grad ratio | 3-step time | Final status |
|---|---:|---|---|---|---|---|---|
| `strongest_scaled.py` | 21,242,304 | PASS | PASS | PASS | PASS / 0.9853x | BLOCKED: missing shards | BUILD/REACH PASS; TIMING BLOCKED |
| `combined_all.py` | 23,021,954 | PASS | PASS | PASS source; mandatory metadata exception recorded | PASS / 1.1128x | BLOCKED: missing shards | BUILD/REACH PASS; TIMING BLOCKED |

The scaled model is a partial approximation: requested MLA substitutes V4 token-compressed attention and static stream mixing substitutes dynamic mHC. The header names all scaling and local choices.
