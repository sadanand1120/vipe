import os
import random

import numpy as np
import torch


_TEMPORARY_DETERMINISM = True


def temporary_determinism_enabled() -> bool:
    return _TEMPORARY_DETERMINISM


def seed_everything(seed: int = 42, temporary_determinism: bool = True) -> None:
    """Seed common RNGs and prefer deterministic backend kernels where PyTorch supports them."""
    global _TEMPORARY_DETERMINISM
    _TEMPORARY_DETERMINISM = bool(temporary_determinism)

    os.environ["PYTHONHASHSEED"] = str(seed)
    if _TEMPORARY_DETERMINISM:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if _TEMPORARY_DETERMINISM:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.use_deterministic_algorithms(False)

    try:
        import open3d as o3d

        o3d.utility.random.seed(seed)
    except Exception:
        pass
