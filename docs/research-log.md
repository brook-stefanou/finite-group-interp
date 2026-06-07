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
