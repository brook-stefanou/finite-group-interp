from finite_group_interp.training.cli import build_config


def test_overrides_are_applied():
    cfg = build_config(["data.group=A4", "optim.epochs=50", "optim.lr=0.005"])
    assert cfg.data.group == "A4"
    assert cfg.optim.epochs == 50
    assert cfg.optim.lr == 0.005


def test_untouched_fields_keep_defaults():
    cfg = build_config(["data.group=A4"])
    assert cfg.model.d_model == 64
    assert cfg.snapshot.event_based is True
    assert cfg.experiment.seed == 0


def test_defaults_with_no_overrides():
    cfg = build_config([])
    assert cfg.data.group == "C8"
    assert cfg.experiment.name == "grok-C8"


def test_run_is_named_after_group():
    cfg = build_config(["data.group=S3"])
    assert cfg.experiment.name == "grok-S3"


def test_explicit_name_is_respected():
    cfg = build_config(["experiment.name=myrun", "data.group=S3"])
    assert cfg.experiment.name == "myrun"


def test_bool_and_float_coercion():
    cfg = build_config(["snapshot.event_based=false", "data.train_frac=0.3"])
    assert cfg.snapshot.event_based is False
    assert cfg.data.train_frac == 0.3
