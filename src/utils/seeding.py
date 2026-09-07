"""Single source of truth for reproducibility. Every entry point calls
set_seed() first, and only this function is allowed to touch the global
RNG state -- otherwise "same seed" claims across M0/M1 stop being true."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
