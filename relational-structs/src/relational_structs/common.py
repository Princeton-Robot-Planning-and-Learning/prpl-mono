"""Common data structures."""

from __future__ import annotations

from typing import (
    Any,
    TypeAlias,
)

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float32]
Action: TypeAlias = Any
