# Research log


## May 31

- Ported existing finite group/representation theory package I had built for use in investigating how group multiplication is learnt.
- Added tests
- Built catalog of all groups with order <20

---

## Jun 1

- Built one layer transformer from scratch with nn.Parameter + einops
- Trainer with logging including event based snapshots of all weights designed to time grokking
- Reproducibility manifest (these use deterministic CPU training which is possible on such small models)
- No LayerNorm, no biases, untied embed/unembed to match Nanda architecture for C113

---

## Jun 2

- Trained on select groups <20 order
  - 4 different groups x 3 weight decays ever run hit train_acc = 1 immediately but never generalised
- Abelian groups test_acc consistently ~2x chance and non-abelian groups test_acc at chance
- Think dataset size too small so it memorises and doesnt generalise (<20 group size gives <= 361 pairs and with train_frac = 0.3 only ~110 training examples)
- Nanda grokking paper has order ~100 which gives ~10,000 pairs and ~3800 training examples
- Pivot to larger groups. Doesn't have to be full classification
- Found it difficult to reproduce Nanda grokking result on C113.
- Realised only difference was my init of weights was much smaller than his.
  - Think that the weight decay on low init weights made it hard for the model to explore loss landscape and find generalised solution
  - After matching fan-in init C113 grokked with test_acc = 0.98 within 10k epochs

---

## Jun 4

- Added causal mask. Doesn't impact previous results since model is 1-layer and loss computer only at the last position
- Fixed bug where MLP layer wasn't being added back to residual stream but was directly applying unembedding to MLP output
- Interesting that still observed grokking on buggy model. Maybe on such simple group MLP was enough?

---

## Jun 7

- Restructure and flatten repo to be more focussed on this project rather than complete portfolio
- Experiment design now of pairs of groups with same character tables.
  - All character level evidence can't tell the groups apart. Any difference in what the networks learn must come from sub-character structure (subgroup lattice vs irrep realization)
  - Dih(104) vs Dic(104)
    -Compare these with C13 ⋊ C8
  - Heisenberg group over F5 vs C25 ⋊ C5
- Built analysis foundation: any checkpoint loads back into a model and forward_with_cache exposes all intermediates
  - cache is recorded inside the real forward pass as a flag, not hooks (not using submodules) or a reimplementation (activations can't drift)
- Realised why C113 is calibration only, not evidence in the irreps-vs-cosets debate: 113 is prime, so there are no proper subgroups — the coset hypothesis is vacuous on this group and can't make a competing prediction. Confirming the irrep signature here validates the measurement tools but adjudicates nothing.
- Expectations for the C113 analysis, written down before running it:
  - Energy concentrates in a small arbitrary set of frequency pairs (~4–6, which ones are seed lottery), near zero in the trivial block (constant functions carry no information about which element a token is)
  - Why several frequencies and not one: logits ≈ Σ_k cos(k(a+b−c)), so used frequencies add coherently at c = a+b and cancel elsewhere — more frequencies sharpen the peak and lower the loss, while weight decay prices each one; the balance lands at a handful
  - Falsification arm: a memorising network would smear energy near-uniformly across all 56 pairs

---

## Jun 8

- Built functional-form fit: regress logits onto ρ(a)ρ(b)ρ(c)⁻¹ matrix elements; full-vs-trace FVE gap = sub-character structure
- Irreps extracted numerically from the regular rep (commutant method; the single-generic-vector idea is wrong for dim ≥ 2)
  - C113 calibration: irrep-formula FVE only 0.55, not the >0.9 I'd assumed
  - But 98% of logit variance is a function of (a+b, c) — it does compute the sum
  - Only 56% depends on a+b−c (translation-invariant); rest = image term (a+b+c) + harmonics
  - Still sparse: 18 of 12769 Fourier modes hold 90% — structured rep-theory, not lookup
  - gap exactly 0 (1-dim ⇒ matrix = trace) — proves the instrument won't invent a gap
  - Lesson: "uses representation theory" ≠ "is the clean textbook formula"
- Built coset side: subgroups/cosets/center + linear probe on residual + coset-direction ablation
  - Probe = nn.Linear + LBFGS (just logistic regression; LBFGS suits small convex fits, no LR to tune)
  - Controls: random-partition null + irrep-feature reference (kept irreps only — all-irrep is vacuous by Peter–Weyl)
  - Both instruments now calibrated on known/planted answers; next is training the pairs

## Jun 9

- First pair runs going. Sweep over Dih(104) vs Dic(104), 3 seeds x 2 weight decays (0.5, 1.0), train_frac 0.4, 80k epochs. Still running (Dic seed 2 + the C13⋊C8 contrast to come).
- Dih groks easily. All 3 seeds, both weight decays. wd 1.0 faster (~9-18k epochs) than wd 0.5 (~25-43k).
- Dic much harder. Only groks at wd 1.0, and ~3-4x slower than Dih (~35-47k vs ~9-18k). At wd 0.5 it never groks
- Same character table for the two, so this learnability gap can't be a character-level thing it has to be sub-character
