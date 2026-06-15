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
- Sweep finished. C13⋊C8 (the third, non-pair group) never groks — all 6 configs stuck at chance (test_acc ~0.01–0.04), pure memorisation, test_loss climbing. Higher weight decay made it worse not better.
- D52 and Dic26 are all dim-1/dim-2, C13⋊C8 has six dim-4 irreps holding 92% of the regular rep.
- So grokking difficulty tracks max irrep dim, not group order (all three are order 104). Small corroboration of the irrep story: the model's pain scales with rep-theoretic complexity, not table size.
- Decision: matched pair comparison stays at wd 1.0 — the only decay where Dic groks across all 3 seeds. wd 0.5 is grok-fragile for Dic (2/3 seeds memorise), so it's not a clean matched setting.

---

## Jun 11

- Expanded to 8 seeds at wd 1.0 (added 3–7). Throttled W&B logging to every 100 epochs — full-res stays in JSONL — after every-step logging lagged the dashboard.
- Learnability is the robust discriminator: Dih groks 8/8, ~15k median; Dic 7/8, ~47k median (~3× slower); the two distributions don't overlap. New: s3 fails to grok even at wd 1.0 (stalls at 0.856 by 80k), so Dic is grok-fragile at the "safe" decay too, not just wd 0.5.
- FVE gap does NOT replicate the Jun 9 separation — that was a high-gap seed. Across all grokked runs: Dih 0.119±0.088 (n=8), Dic 0.048±0.045 (n=7). Both positive (every run uses full rank-4 in its dim-2 blocks), Dih>Dic in 5/7 seeds, but error bars overlap — too seed-noisy to discriminate on its own. Corroborates the irrep account, doesn't carry it.
- Coset excess_over_irrep ~0 and indistinguishable between groups (Dih −0.034±0.172, Dic −0.035±0.104 over all normal subgroups × seeds) — cosets don't separate the pair. Correction to the earlier "no positive excess anywhere": D52-s3 hits +0.54 on a few subgroups, but that's a keep-set under-counting artifact (energy>2×baseline missed a used irrep → weak irrep reference), on the dihedral side and one seed. Honest claim: no group-level coset signal, not zero everywhere.

---

## Jun 14

- Final 38-seed sweep, wd 1.0 (f 0.4, 80k ep), both groups; the 27 where both grok carry the matrix/coset contrasts. Learnability split sharpens: Dih 35/38 (~20k, 3 near-threshold misses, none stuck); Dic 29/38 (~40k, ~2× slower, 6 of 9 misses stuck in pure memorisation, low 0.02); Dic still never groks at wd 0.5. Different failure modes, not just speed — and character-identical, so it's sub-character, in training not the weights.
- R² gap (renamed from FVE) a real null at n=27: Dih 0.074±0.064, Dic 0.055±0.054, Welch p=0.25. Jun 9 "separation" was one high-gap seed; both fill full rank in dim-2 blocks (kills the "real uses half the rank" prediction). Coset excess ≤0 too (Dih −0.055, Dic −0.044 over 7 normal subgroups × 27 seeds) — irrep control matches the naive probe, no signal on top.
- Dim-5 prereq before building the order-125 pair: does a dim-5 group grok at all? Heisenberg/F5 (order 125, four 5-dim irreps), base wd1.0/f0.5 → 1M ep, train 1.0 but test plateaus ~0.11 (chance 0.008). Probes (wd 2.0, f 0.7, both, 80k each) don't rescue it — peaks 0.10/0.12/0.15, both-levers still climbing. No grok within budget, not "cannot". Ladder: dim-1 fast → dim-2 easy Dih/hard Dic → dim-4 (C13⋊C8) no-grok 80k → dim-5 no-grok 1M; tracks rep-theoretic complexity, not order. Parks the C25⋊C5 pair until a dim-5 groks.
- Finalized report 02 + README at 38 seeds.

---

## Jun 15

- Architecture confound (coset lit is FC, irrep lit transformer): retrained the pair on a 1-hidden-layer FC net (shared embed → concat → ReLU, no biases), 6 seeds/group, same wd. All three findings survive — learnability ~3× split (Dih ~5,600 ep, Dic ~16,100), R² gap null (0.021 vs 0.013, p=0.18), coset ≈0 (Dih +0.05, Dic −0.04). One difference: every Dic seed groks on FC, so the memorisation plateau (6/38 on the transformer) is transformer-specific — here it's purely speed.
- Coset-null on the very architecture the coset account came from = the most demanding control. Folded into report 02 + README (+ fc- figures); also fixed the C113 headline to 99.77%.
