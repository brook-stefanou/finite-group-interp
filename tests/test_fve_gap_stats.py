"""Stats backing for report 02's FVE-gap section.

Parses the matrix-level GAP lines out of `compare_pairs.py` output and runs a
Welch t-test between the two groups -- pure text/arithmetic, no model loading, so
it never contends with a training run. The point: when the larger seed set lands,
re-run compare_pairs once, pipe the log through here, drop mean+/-std + p into the
report stub.
"""

from same_character_table_interp.analysis.fve_gap_stats import (
    betainc_reg,
    gaps_from_csv,
    parse_fve_gaps,
    summarize,
    welch_ttest,
)

# A miniature of real compare_pairs --coset output (two matched seeds).
SAMPLE = """
===================================================================================
MATRIX-LEVEL CONTRAST (matched seed x weight-decay; >=2 groups grokked)
===================================================================================

########## matched setting: s1 wd1.0 ##########

=== 2026-06-09_024645_pair-D52-s1-wd1.0-f0.4  (order 104, checkpoint epoch 8826) ===
  kept blocks (W_E energy > 2x baseline): [0, 3, 10]
  functional-form FVE: full=0.511 trace=0.505  GAP=0.006

=== 2026-06-09_041411_pair-Dic26-s1-wd1.0-f0.4  (order 104, checkpoint epoch 35310) ===
  functional-form FVE: full=0.387 trace=0.360  GAP=0.028

########## matched setting: s2 wd1.0 ##########

=== 2026-06-09_030218_pair-D52-s2-wd1.0-f0.4  (order 104, checkpoint epoch 18097) ===
  functional-form FVE: full=0.415 trace=0.398  GAP=0.017

=== 2026-06-09_045403_pair-Dic26-s2-wd1.0-f0.4  (order 104, checkpoint epoch 32317) ===
  functional-form FVE: full=0.428 trace=0.424  GAP=0.004
"""


def test_parse_groups_seeds_and_gaps():
    parsed = parse_fve_gaps(SAMPLE)
    assert set(parsed) == {"D52", "Dic26"}
    d52 = sorted(parsed["D52"], key=lambda r: r["seed"])
    assert [r["seed"] for r in d52] == [1, 2]
    assert [r["wd"] for r in d52] == ["1.0", "1.0"]
    assert [r["gap"] for r in d52] == [0.006, 0.017]
    assert sorted(r["gap"] for r in parsed["Dic26"]) == [0.004, 0.028]


def test_parse_ignores_learnability_table_and_noise():
    # Lines without a pair- header or a GAP must not produce records.
    noise = "run  grok@  final_acc\n2026-06-09_024645_pair-D52-s1-wd1.0-f0.4  8822  0.9903\n"
    assert parse_fve_gaps(noise) == {}


def test_betainc_reg_endpoints_and_symmetry():
    assert betainc_reg(2.0, 3.0, 0.0) == 0.0
    assert betainc_reg(2.0, 3.0, 1.0) == 1.0
    # I_0.5(a, a) = 0.5 by symmetry for equal parameters.
    assert abs(betainc_reg(2.5, 2.5, 0.5) - 0.5) < 1e-9


def test_welch_matches_known_value():
    # a, b each n=5, equal sample variance 2.5, means 3 and 5 -> t=-2, dof=8.
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [3.0, 4.0, 5.0, 6.0, 7.0]
    res = welch_ttest(a, b)
    assert abs(res["t"] - (-2.0)) < 1e-9
    assert abs(res["dof"] - 8.0) < 1e-9
    # two-sided p for |t|=2 at 8 dof is ~0.0805 (t-table).
    assert abs(res["p_two_sided"] - 0.0805) < 2e-3


def test_welch_zero_t_gives_p_one():
    res = welch_ttest([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert abs(res["t"]) < 1e-12
    assert abs(res["p_two_sided"] - 1.0) < 1e-9


CSV = (
    "run_id,group,seed,wd,grokked,grok_epoch,energy_concentration,gap,coset_max_excess_over_irrep,label\n"
    "x-D52-s1,D52,1,1.0,True,8822,0.9,0.006,0.0,irrep-consistent\n"
    "x-Dic26-s1,Dic26,1,1.0,True,35292,0.9,0.028,0.0,irrep-consistent\n"
)


def test_gaps_from_csv_groups_by_group() -> None:
    parsed = gaps_from_csv(CSV)
    assert parsed["D52"][0]["gap"] == 0.006
    assert parsed["Dic26"][0]["gap"] == 0.028


def test_summarize_reports_both_groups_and_p():
    out = summarize(parse_fve_gaps(SAMPLE))
    assert "D52" in out and "Dic26" in out
    assert "n=" in out and "mean=" in out
    assert "Welch" in out and "p=" in out
    assert "nan" not in out.lower()  # n>=2 per group, so std/p must be real numbers
