"""Scoped provider fixtures must not depend on or rewrite the user's .env."""
import pytest

import config


@pytest.fixture
def use_test_model(monkeypatch):
    def select(model):
        # GENERATION_CHAINS is derived at config import time. A fixture changing
        # only SEEDANCE_MODEL otherwise creates a contradictory provider setup.
        monkeypatch.setattr(config, 'SEEDANCE_MODEL', model)
        monkeypatch.setattr(config, 'GENERATION_CHAINS', {
            resolution: [{**profile, 'model': model} for profile in profiles]
            for resolution, profiles in config.GENERATION_CHAINS.items()
        })
    return select
