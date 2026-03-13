"""Utility for loading PyTorch Hub models without conflicting with the local models/ package."""
import sys
import torch


def safe_hub_load(repo, model, *args, **kwargs):
    """Load a model from torch.hub without conflicting with the local ``models`` package.

    When this project's root is on ``sys.path``, Python caches the local
    ``models/`` package under the key ``"models"`` in ``sys.modules``.
    torch.hub's ``hubconf.py`` then resolves ``from models import ...`` to our
    package instead of the hub repo's own ``models/`` directory, causing an
    ``ImportError``.  This helper temporarily removes those cached entries,
    delegates to ``torch.hub.load``, and restores them afterwards.
    """
    saved = {k: v for k, v in list(sys.modules.items())
             if k == 'models' or k.startswith('models.')}
    for k in saved:
        del sys.modules[k]
    try:
        return torch.hub.load(repo, model, *args, **kwargs)
    finally:
        sys.modules.update(saved)
