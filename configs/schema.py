from pydantic import BaseModel, validator, Field
from typing import Dict, Any, Optional, Union, List, Sequence
from dataloaders.net import DISCOVERY_METHODS
import importlib
import torch.distributions as dist
from dataloaders.runs import AlignerSpec, Aligner, PM4pyAligner, TraceSampler
from dataloaders.net import VariantRandomDistributionSampler
from pm4py.algo.conformance.alignments.petri_net.algorithm import Variants
from util.rng import RNG


class DistributionConfig(BaseModel):
    type: str
    args: Dict[str, Any]

    def build(self) -> dist.Distribution:
        module, cls_name = self.type.rsplit(".", 1)
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)

        if not issubclass(cls, dist.Distribution):
            raise TypeError(f"{self.type} is not a torch.distribution class")

        return cls(**self.args)


class SamplerConfig(BaseModel):
    name: str
    n_subsets: int
    min_len_subset: int
    max_len_subset: int
    len_distribution: DistributionConfig
    freq_distribution: DistributionConfig

    @validator("max_len_subset")
    def len_check(cls, v, values):
        if "min_len_subset" in values and v < values["min_len_subset"]:
            raise ValueError("max_len_subset must be >= min_len_subset")
        return v

    def build(self, rng: RNG) -> VariantRandomDistributionSampler:
        len_dist = self.len_distribution.build()
        freq_dist = self.freq_distribution.build()
        sampler = VariantRandomDistributionSampler(
            n_subsets=self.n_subsets,
            min_len_subset=self.min_len_subset,
            max_len_subset=self.max_len_subset,
            len_distribution=len_dist,
            freq_distribution=freq_dist,
            seed=rng.get_seed(),
        )
        return sampler


class DiscoveryConfig(BaseModel):
    workers: Optional[int] = None
    methods: Union[str, List[str]]
    params: Dict[str, Any]
    sampler: SamplerConfig

    @validator("methods")
    def validate_methods(cls, v):
        # Preset name?
        if isinstance(v, str) and v in DISCOVERY_METHODS.__members__:
            return list(DISCOVERY_METHODS[v].value.keys())

        # Single discovery method?
        if isinstance(v, str) and v in DISCOVERY_METHODS.ALL.value.keys():
            return [v]

        # List of methods?
        if isinstance(v, list):
            missing = set(v) - set(DISCOVERY_METHODS.ALL.value.keys())
            if missing:
                raise ValueError(f"Unknown discovery methods: {missing}")
            return v

        raise ValueError(f"Invalid discovery config: {v}")

    def resolve(self) -> Dict[str, Any]:
        """Map names → callables."""
        result = {}
        for name, func in DISCOVERY_METHODS.ALL.value.items():
            if name in self.methods:
                result[name] = func
        return result


class TraceSamplerConfig(BaseModel):
    type: str

    def build(self) -> TraceSampler.__class__:
        class_name = self.type.rsplit('.', 1)
        if len(class_name) < 2:
            module_name = "dataloaders.runs"
            class_name = class_name[0]
        else:
            module_name, class_name = class_name
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls


class AlignmentConfig(BaseModel):
    runs: int = Field(gt=0)
    workers: int = Field(ge=0)
    variants: Union[str, list[str]]
    sampler: TraceSamplerConfig

    @validator("variants", pre=True)
    def validate_variants(cls, v):
        if not isinstance(v, (list, str)):
            raise ValueError("Alignment variants must be a list or string")
        if isinstance(v, str):
            # we need to check if it's in AlignerSpec
            if v not in AlignerSpec.__members__:
                raise ValueError(
                    f"Unknown alignment spec variant: {v}; Available options: {list(AlignerSpec.__members__.keys())}"
                )
            return v
        for variant in v:
            if variant not in Variants.__members__:
                raise ValueError(f"Unknown alignment variant: {variant}")
        return v

    def resolve(self) -> Sequence[Aligner]:
        if isinstance(self.variants, str):
            return AlignerSpec[self.variants].value
        return [PM4pyAligner(Variants[variant]) for variant in self.variants]


class PipelineConfig(BaseModel):
    seed: int = 1
    log_path: Optional[str] = None
    discovery: DiscoveryConfig
    alignment: AlignmentConfig
