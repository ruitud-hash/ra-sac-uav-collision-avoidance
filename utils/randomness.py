"""Random-state isolation helpers for training and evaluation."""

from __future__ import annotations

from functools import wraps
import random
from typing import Any, Callable, TypeVar, cast

import numpy as np
import torch


F = TypeVar("F", bound=Callable[..., Any])


def preserve_global_rng_state(function: F) -> F:
    """Run a function without advancing Python, NumPy, or Torch global RNGs."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            return function(*args, **kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    return cast(F, wrapped)
