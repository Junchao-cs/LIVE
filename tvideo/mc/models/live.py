"""Public LIVE model API."""

from tvideo.mc.models.df_flow import DiffusionForcing


class LIVE(DiffusionForcing):
    """LIVE post-training model for cycle-consistent video world modeling."""


__all__ = ["LIVE"]
