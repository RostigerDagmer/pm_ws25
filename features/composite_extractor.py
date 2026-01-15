from typing import Dict, List, Union
from pm4py.objects.petri_net.obj import PetriNet, Marking
from features.base_extractor import BaseFeatureExtractor
from features.model_extractor import ModelFeatureExtractor
from features.trace_extractor import TraceFeatureExtractor
from features.state_space_size_extractor import StateSpaceSizeExtractor
from features.token_replay_extractor import TokenReplayFitnessExtractor


class CompositeFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts features from both model and trace, including interaction features.

    Combines features from ModelFeatureExtractor and TraceFeatureExtractor,
    then adds interaction features that relate model and trace characteristics.
    """

    def __init__(self, use_cache: bool = True):
        super().__init__(use_cache=use_cache)
        # set use_cache=False for sub-extractors to avoid double caching
        self.model_extractor = ModelFeatureExtractor(use_cache=False)
        self.trace_extractor = TraceFeatureExtractor(use_cache=False)
        self.token_replay_extractor = TokenReplayFitnessExtractor(
            use_cache=False
        )

    def _compute_cache_key(
        self,
        petri_net: PetriNet,
        petri_net_im: Marking,
        petri_net_fm: Marking,
        trace_net: PetriNet,
        trace_net_im: Marking,
        trace_net_fm: Marking,
    ):
        """Use tuple of both net hashes as cache key."""
        return (hash(petri_net), hash(trace_net))

    @property
    def feature_names(self) -> List[str]:
        return (
            self.model_extractor.feature_names
            + self.trace_extractor.feature_names
            + [
                'interaction_n_activity_present_in_model',
                'interaction_n_activity_not_in_model',
                'interaction_activity_coverage_ratio',
            ]
            + self.token_replay_extractor.feature_names
        )

    def _extract_features_internal(
        self,
        petri_net: PetriNet,
        petri_net_im: Marking,
        petri_net_fm: Marking,
        trace_net: PetriNet,
        trace_net_im: Marking,
        trace_net_fm: Marking,
    ) -> Dict[str, float]:
        """Extract model, trace, and interaction features."""
        model_features = self.model_extractor.extract(
            petri_net, petri_net_im, petri_net_fm, return_as_dict=True
        )
        trace_features = self.trace_extractor.extract(
            trace_net, trace_net_im, trace_net_fm, return_as_dict=True
        )
        interaction_features = self._extract_interactions(
            petri_net, petri_net_im, petri_net_fm, trace_net
        )

        token_replay_features = self.token_replay_extractor.extract(
            petri_net,
            petri_net_im,
            petri_net_fm,
            trace_net,
            trace_net_im,
            trace_net_fm,
            return_as_dict=True,
        )

        return {
            **model_features,
            **trace_features,
            **interaction_features,
            **token_replay_features,
        }

    def _extract_interactions(
        self,
        model_net: PetriNet,
        model_im: Marking,
        model_fm: Marking,
        trace_net: PetriNet,
    ) -> Dict[str, float]:
        """Extract interaction features between model and trace."""
        model_labels = {
            t.label for t in model_net.transitions if t.label is not None
        }
        trace_labels = [
            t.label for t in trace_net.transitions if t.label is not None
        ]

        present_in_model = sum(
            1 for label in trace_labels if label in model_labels
        )
        not_in_model = sum(
            1 for label in trace_labels if label not in model_labels
        )

        trace_length = len(trace_labels)
        coverage_ratio = (
            present_in_model / trace_length if trace_length > 0 else 0.0
        )

        return {
            'interaction_n_activity_present_in_model': present_in_model,
            'interaction_n_activity_not_in_model': not_in_model,
            'interaction_activity_coverage_ratio': coverage_ratio,
        }

    def _extract_features_batch(
        self,
        petri_net: PetriNet,
        petri_net_im: Marking,
        petri_net_fm: Marking,
        trace_nets: list[tuple[PetriNet, Marking, Marking]],
    ) -> list[Dict[str, float]]:
        """Extract features for a batch of traces."""
        model_features = self.model_extractor.extract(
            petri_net, petri_net_im, petri_net_fm, return_as_dict=True
        )
        trace_features = [
            self.trace_extractor.extract(
                trace_net, trace_net_im, trace_net_fm, return_as_dict=True
            )
            for trace_net, trace_net_im, trace_net_fm in trace_nets
        ]
        interaction_features = [
            self._extract_interactions(
                petri_net, petri_net_im, petri_net_fm, trace_net
            )
            for trace_net, _, _ in trace_nets
        ]
        token_replay_features = [
            self.token_replay_extractor.extract(
                petri_net,
                petri_net_im,
                petri_net_fm,
                trace_net,
                trace_net_im,
                trace_net_fm,
                return_as_dict=True,
            )
            for trace_net, trace_net_im, trace_net_fm in trace_nets
        ]

        return [
            {
                **model_features,
                **trace_features[i],
                **interaction_features[i],
                **token_replay_features[i],
            }
            for i in range(len(trace_nets))
        ]
