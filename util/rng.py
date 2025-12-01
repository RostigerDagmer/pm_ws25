import random

import numpy as np
import torch


class RNG:
    """Random number generator manager."""

    _seed: int | None = None
    _np_generator: np.random.Generator | None = None

    @staticmethod
    def initialize(seed: int = 1) -> None:
        """Set global RNG seed for reproducibility."""
        if RNG._seed is not None:
            msg = "RNG is already initialized. Multiple initializations are not allowed."
            raise RuntimeError(msg)

        RNG._seed = seed

        random.seed(seed)

        # NumPy
        RNG._np_generator = np.random.default_rng(seed)

        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    @staticmethod
    def get_seed() -> int:
        """Get the RNG seed used for initialization."""
        if RNG._seed is None:
            msg = "RNG is not initialized. Call RNG.initialize(...) first."
            raise RuntimeError(msg)
        return RNG._seed

    @staticmethod
    def np_generator() -> np.random.Generator:
        """Get the NumPy generator initialized with the RNG seed."""
        if RNG._np_generator is None:
            msg = "RNG is not initialized. Call RNG.initialize(...) first."
            raise RuntimeError(msg)
        return RNG._np_generator

    @staticmethod
    def torch_generator(device: torch.device | str = "cpu") -> torch.Generator:
        """Get a PyTorch generator initialized with the RNG seed."""
        if RNG._seed is None:
            msg = "RNG is not initialized. Call RNG.initialize(...) first."
            raise RuntimeError(msg)
        gen = torch.Generator(device=device)
        gen.manual_seed(RNG._seed)
        return gen
