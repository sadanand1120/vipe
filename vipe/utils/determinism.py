import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed common RNGs without forcing slower deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import open3d as o3d

        o3d.utility.random.seed(seed)
    except Exception:
        pass
