from typing import Any
from typing import Union
from typing import Sequence
from typing import Optional
import torch
from torch.types import _int, SymInt
from dataclasses import dataclass


@dataclass(frozen=True)
class CategoricalSpec:
    probs: list[float]


@dataclass(frozen=True)
class PoissonSpec:
    rate: float


@dataclass(frozen=True)
class BernoulliDepthLinearSpec:
    base: float
    slope: float


@dataclass(frozen=True)
class NormalSpec:
    mean: float
    std: float


@dataclass(frozen=True)
class ExponentialSpec:
    rate: float


class WrappedCategorical:
    def __init__(self, probs):
        self.probs = torch.tensor(probs, dtype=torch.float)

    def sample(
        self, size: Optional[Sequence[_int | SymInt]] = None, generator=None
    ):
        return self.probs.multinomial(
            size if size else 1, generator=generator
        ).squeeze()


class WrappedPoisson:
    def __init__(self, rate):
        self.rate = torch.tensor(rate, dtype=torch.float)

    def sample(
        self, size: Optional[Sequence[_int | SymInt]] = None, generator=None
    ):
        return torch.poisson(
            torch.ones(size if size else [1]) * self.rate, generator=generator
        )


class WrappedBernoulli:
    def __init__(self, probs):
        self.probs = torch.tensor(probs, dtype=torch.float)

    def sample(
        self, size: Optional[Sequence[_int | SymInt]] = None, generator=None
    ):
        return torch.zeros(
            size if size is not None else [1], dtype=torch.float
        ).bernoulli_(self.probs, generator=generator)


class WrappedNormal:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float)
        self.std = torch.tensor(std, dtype=torch.float)

    def sample(
        self, size: Optional[Sequence[_int | SymInt]] = None, generator=None
    ):
        return torch.zeros(
            size if size is not None else [1], dtype=torch.float
        ).normal_(self.mean, self.std, generator=generator)


class WrappedExponential:
    def __init__(self, rate):
        self.rate = torch.tensor(rate, dtype=torch.float)

    def sample(
        self, size: Optional[Sequence[_int | SymInt]] = None, generator=None
    ):
        return torch.zeros(
            size if size is not None else [1], dtype=torch.float
        ).exponential_(self.rate, generator=generator)


DistParam = Union[CategoricalSpec, PoissonSpec, BernoulliDepthLinearSpec]


def make_distribution(spec: DistParam, *, depth=None):
    if isinstance(spec, CategoricalSpec):
        return WrappedCategorical(spec.probs)
    if isinstance(spec, PoissonSpec):
        return WrappedPoisson(spec.rate)
    if isinstance(spec, BernoulliDepthLinearSpec):
        p = spec.base + spec.slope * depth
        return WrappedBernoulli(min(max(p, 0), 1))
    if isinstance(spec, NormalSpec):
        return WrappedNormal(spec.mean, spec.std)
    if isinstance(spec, ExponentialSpec):
        return WrappedExponential(spec.rate)
    raise TypeError(spec)


def deserialize(spec: dict[str, Any]):
    if spec["type"] == "CategoricalSpec":
        return CategoricalSpec(**spec["args"])
    if spec["type"] == "PoissonSpec":
        return PoissonSpec(**spec["args"])
    if spec["type"] == "BernoulliDepthLinearSpec":
        return BernoulliDepthLinearSpec(**spec["args"])
    if spec["type"] == "NormalSpec":
        return NormalSpec(**spec["args"])
    if spec["type"] == "ExponentialSpec":
        return ExponentialSpec(**spec["args"])
    raise TypeError(spec)
