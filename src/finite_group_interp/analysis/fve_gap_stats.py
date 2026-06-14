"""Aggregate the matrix-level FVE gaps from ``compare_pairs.py`` output and run a
Welch t-test between the two groups.

The FVE gap (full-matrix FVE minus trace-only FVE) is the sub-character quantity
report 02 turns on. ``compare_pairs`` prints one gap per matched run; this module
parses those lines, summarises each group's gap distribution, and tests whether the
two groups differ -- all from text, so it never re-loads a checkpoint or competes
with a running sweep. Welch's t-test (unequal variances) is implemented in pure
Python: the p-value comes from the Student-t survival function via the regularised
incomplete beta function, since scipy is not a dependency.
"""

from __future__ import annotations

import csv as _csv
import io
import math
import re
import statistics
from typing import TypedDict


class GapRecord(TypedDict):
    seed: int
    wd: str
    gap: float


# `=== 2026-06-09_..._pair-D52-s1-wd1.0-f0.4  (order 104, ...) ===`
_HEADER = re.compile(r"pair-([A-Za-z0-9]+)-s(\d+)-wd([\d.]+)")
# `  functional-form FVE: full=0.511 trace=0.505  GAP=0.006`
_GAP = re.compile(r"GAP=(-?[\d.]+)")


def parse_fve_gaps(text: str) -> dict[str, list[GapRecord]]:
    """Map each group token to a list of ``{"seed", "wd", "gap"}`` records.

    Walks the lines in order: a ``pair-<group>-s<seed>-wd<wd>`` header arms the next
    ``GAP=`` line, which is attributed to that run. Lines that are neither (the
    learnability table, block listings, noise) are ignored.
    """
    out: dict[str, list[GapRecord]] = {}
    group: str | None = None
    seed: int | None = None
    wd: str | None = None
    for line in text.splitlines():
        header = _HEADER.search(line)
        if header:
            group, seed, wd = header.group(1), int(header.group(2)), header.group(3)
            continue
        gap = _GAP.search(line)
        if gap and group is not None and seed is not None and wd is not None:
            out.setdefault(group, []).append({"seed": seed, "wd": wd, "gap": float(gap.group(1))})
            group = seed = wd = None  # consume; one gap per header
    return out


def gaps_from_csv(text: str) -> dict[str, list[GapRecord]]:
    """Read group -> gap records from compare.py's comparison.csv."""
    out: dict[str, list[GapRecord]] = {}
    for row in _csv.DictReader(io.StringIO(text)):
        out.setdefault(row["group"], []).append(
            {"seed": int(row["seed"]), "wd": row["wd"], "gap": float(row["gap"])}
        )
    return out


def _betacf(a: float, b: float, x: float, iters: int = 200) -> float:
    """Continued fraction for the incomplete beta (Lentz's algorithm)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_sf_two_sided(t: float, dof: float) -> float:
    """Two-sided p-value P(|T| > |t|) for a Student-t with `dof` degrees of freedom."""
    if dof <= 0:
        return float("nan")
    x = dof / (dof + t * t)
    return betainc_reg(dof / 2.0, 0.5, x)


def welch_ttest(a: list[float], b: list[float]) -> dict[str, float]:
    """Welch's unequal-variance t-test. Returns t, dof (Welch-Satterthwaite), p."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("each group needs at least 2 observations for a t-test")
    va, vb = statistics.variance(a), statistics.variance(b)
    sa, sb = va / na, vb / nb
    se2 = sa + sb
    if se2 == 0.0:
        return {"t": 0.0, "dof": float(na + nb - 2), "p_two_sided": 1.0}
    t = (statistics.fmean(a) - statistics.fmean(b)) / math.sqrt(se2)
    dof = se2**2 / (sa**2 / (na - 1) + sb**2 / (nb - 1))
    return {"t": t, "dof": dof, "p_two_sided": _student_t_sf_two_sided(t, dof)}


def summarize(parsed: dict[str, list[GapRecord]]) -> str:
    """A printable per-group summary plus a Welch t-test when exactly two groups."""
    lines = ["FVE gap (full - trace) by group:", ""]
    series: dict[str, list[float]] = {}
    for group in sorted(parsed):
        gaps = [r["gap"] for r in parsed[group]]
        series[group] = gaps
        mean = statistics.fmean(gaps)
        sd = statistics.stdev(gaps) if len(gaps) > 1 else float("nan")
        rng = f"[{min(gaps):.3f}, {max(gaps):.3f}]"
        lines.append(f"  {group:8s} n={len(gaps):2d}  mean={mean:.3f}  std={sd:.3f}  range={rng}")

    groups = sorted(series)
    if len(groups) == 2 and all(len(series[g]) >= 2 for g in groups):
        ga, gb = groups
        res = welch_ttest(series[ga], series[gb])
        lines += [
            "",
            f"Welch t-test ({ga} vs {gb}): "
            f"t={res['t']:.3f}  dof={res['dof']:.1f}  p={res['p_two_sided']:.4f}",
            f"  => {'significant' if res['p_two_sided'] < 0.05 else 'NOT significant'} at alpha=0.05",
        ]
    return "\n".join(lines)
