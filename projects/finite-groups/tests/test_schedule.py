from finite_groups.grokking.config import SnapshotConfig
from finite_groups.grokking.schedule import should_snapshot


def test_initial_step_is_always_snapshotted():
    assert should_snapshot(0, SnapshotConfig())


def test_dense_powers_of_two_in_early_region():
    cfg = SnapshotConfig(log_dense_until=1024, interval=1000)
    for step in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        assert should_snapshot(step, cfg), step


def test_non_power_of_two_in_dense_region_is_skipped():
    cfg = SnapshotConfig(log_dense_until=1024, interval=1000)
    assert not should_snapshot(3, cfg)
    assert not should_snapshot(100, cfg)


def test_periodic_snapshots_after_dense_region():
    cfg = SnapshotConfig(log_dense_until=1024, interval=1000)
    assert should_snapshot(2000, cfg)
    assert should_snapshot(3000, cfg)
    assert not should_snapshot(2500, cfg)


def test_disabled_never_snapshots():
    cfg = SnapshotConfig(enabled=False)
    assert not should_snapshot(0, cfg)
    assert not should_snapshot(1, cfg)
    assert not should_snapshot(1000, cfg)


def test_negative_step_never_snapshots():
    assert not should_snapshot(-1, SnapshotConfig())
