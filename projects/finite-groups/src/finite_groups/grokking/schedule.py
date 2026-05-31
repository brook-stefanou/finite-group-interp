"""Deterministic snapshot schedule: when to checkpoint weights during training.

Dense early (powers of two, to catch fast initial learning), then periodic.
Grokking transitions are slow, so this keeps a few hundred snapshots over a long
run instead of one per step. Event-based re-densification around the test-loss
drop is a trainer-level concern layered on top of this baseline predicate.
"""

from finite_groups.grokking.config import SnapshotConfig


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def should_snapshot(step: int, config: SnapshotConfig) -> bool:
    """Whether to snapshot weights at ``step`` under the given schedule."""
    if not config.enabled or step < 0:
        return False
    if step == 0:
        return True
    if step <= config.log_dense_until and _is_power_of_two(step):
        return True
    return step % config.interval == 0
