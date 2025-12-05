from util.distributions import DistParam
from dataloaders.unique_net import UniqueProcessModelDataset
from dataloaders.net import ProcessModelDataset
from pydantic import BaseModel, field_validator, Field, ValidationInfo
from typing import Dict, Any, Optional, Union, List, Sequence
from dataloaders.net import DISCOVERY_METHODS
import importlib
import torch.distributions as dist
from dataloaders.runs import AlignerSpec, Aligner, PM4pyAligner, TraceSampler
from dataloaders.net import VariantRandomDistributionSampler
from pm4py.algo.conformance.alignments.petri_net.algorithm import Variants
from util.rng import RNG
from deduplication.deduplicator import DeduplicationConfig


class DistributionConfig(BaseModel):
    type: str
    args: Dict[str, Any]

    def build(self) -> DistParam:
        cls_name = (
            f"{self.type}Spec" if not self.type.endswith("Spec") else self.type
        )
        mod = importlib.import_module("util.distributions")
        try:
            cls = getattr(mod, cls_name)
        except AttributeError:
            raise ValueError(f"Unknown distribution type: {self.type}")

        return cls(**self.args)


class SamplerConfig(BaseModel):
    name: str
    n_subsets: int
    min_len_subset: int
    max_len_subset: int
    len_distribution: DistributionConfig
    freq_distribution: DistributionConfig

    @field_validator("max_len_subset")
    def len_check(cls, v, values: ValidationInfo):
        if (
            "min_len_subset" in values.data
            and v < values.data["min_len_subset"]
        ):
            raise ValueError("max_len_subset must be >= min_len_subset")
        return v

    def build(self) -> VariantRandomDistributionSampler:
        len_dist = self.len_distribution.build()
        freq_dist = self.freq_distribution.build()
        sampler = VariantRandomDistributionSampler(
            seed=RNG.get_seed(),
            n_subsets=self.n_subsets,
            min_len_subset=self.min_len_subset,
            max_len_subset=self.max_len_subset,
            len_distribution=len_dist,
            freq_distribution=freq_dist,
        )
        return sampler


class DiscoveryConfig(BaseModel):
    workers: Optional[int] = None
    methods: Union[str, List[str]]
    params: Dict[str, Any]
    samplers: list[SamplerConfig]

    @field_validator("methods")
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


class SliceType(BaseModel):
    from_: int = Field(alias="from", ge=0)
    to: int = Field(gt=0)

    @field_validator("to")
    def check_to(cls, v, values: ValidationInfo):
        if "from_" in values.data and v <= values.data["from_"]:
            raise ValueError("'to' must be greater than 'from'")
        return v


class TraceSamplerConfig(BaseModel):
    type: str
    slice: Optional[SliceType] = None
    args: Dict[str, Any] = Field(default_factory=dict)

    def build(
        self,
        ds: Union[ProcessModelDataset, UniqueProcessModelDataset],
    ) -> TraceSampler:
        class_name = self.type.rsplit('.', 1)
        if len(class_name) < 2:
            module_name = "dataloaders.runs"
            class_name = class_name[0]
        else:
            module_name, class_name = class_name
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return (
            cls(
                seed=RNG.get_seed(),
                ds=ds,
                slice=range(self.slice.from_, self.slice.to),
                **self.args,
            )
            if self.slice
            else cls(seed=RNG.get_seed(), ds=ds, **self.args)
        )


class AlignmentConfig(BaseModel):
    runs: int = Field(gt=0)
    cache_path: Optional[str] = None
    workers: int = Field(ge=0)
    write_batch_size: int = Field(gt=0)
    variants: Union[str, list[str]]
    sampler: TraceSamplerConfig

    @field_validator("variants")
    def validate_variants(cls, v: Any):
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


class DedupeConfig(BaseModel):
    config: DeduplicationConfig = DeduplicationConfig()
    force_recompute: bool = False

    @field_validator("config")
    def validate_config(cls, v: dict[str, Any] | DeduplicationConfig):
        if isinstance(v, dict):
            return DeduplicationConfig(**v)
        return v


class PipelineConfig(BaseModel):
    seed: int = 1
    log_path: Optional[str] = None
    discovery: DiscoveryConfig
    deduplication: Optional[DedupeConfig] = None
    alignment: AlignmentConfig
