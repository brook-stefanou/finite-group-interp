import os

# use_wandb now defaults to True, and several trainer tests call fit(). Disable
# W&B entirely during the test suite so it never opens a network session or
# writes run files -- wandb.init/log/finish all become no-ops under this mode.
os.environ["WANDB_MODE"] = "disabled"
