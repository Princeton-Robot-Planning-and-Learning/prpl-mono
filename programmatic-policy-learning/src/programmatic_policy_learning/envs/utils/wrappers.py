"""Environment wrappers and utilities for common issues."""

from typing import Callable

import numpy as np
from gymnasium import spaces


def patch_box_float32() -> Callable[..., None]:
    """Resolves Gymnasium Box precision warnings by patching space creation to
    use float32 dtypes from the start, eliminating the need for runtime casting
    that triggers "precision lowered" warnings from environment libraries like
    PRBench (PRBench creates box spaces with float64 bounds).

    Returns:
        The original Box.__init__ method for restoration.
    """

    original_box_init = spaces.Box.__init__

    def patched_box_init(self, low, high, shape=None, dtype=np.float32, seed=None):
        # Convert bounds to float32 if they're float64 to avoid warnings
        if hasattr(low, "dtype") and low.dtype == np.float64:
            low = low.astype(np.float32)
        if hasattr(high, "dtype") and high.dtype == np.float64:
            high = high.astype(np.float32)

        # Force dtype to float32 for floating point types
        if dtype == np.float64:
            dtype = np.float32

        return original_box_init(self, low, high, shape, dtype, seed)

    spaces.Box.__init__ = patched_box_init  # type: ignore
    return original_box_init
