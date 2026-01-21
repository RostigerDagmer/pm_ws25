import logging
from typing import Dict, List
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.tokenreplay import algorithm as executor
from pm4py.algo.conformance.tokenreplay.variants import token_replay
from pm4py.util import xes_constants as xes_util

from features.base_extractor import BaseFeatureExtractor


class TokenReplayFitnessExtractor(BaseFeatureExtractor):
    """
    Extracts token replay fitness features between a Petri net model and a trace.

    Features include:
    - trace_is_fit: Boolean indicating if the trace perfectly fits the model
    - trace_fitness: Fitness value for the individual trace (0.0 to 1.0)
    - missing_tokens: Number of missing tokens during replay
    - consumed_tokens: Number of consumed tokens during replay
    - remaining_tokens: Number of remaining tokens after replay
    - produced_tokens: Number of produced tokens during replay
    """

    @property
    def feature_names(self) -> List[str]:
        return [
            'token_replay_trace_is_fit',
            'token_replay_trace_fitness',
            'token_replay_missing_tokens',
            'token_replay_consumed_tokens',
            'token_replay_remaining_tokens',
            'token_replay_produced_tokens',
        ]

    def _extract_features_internal(
        self,
        petri_net: PetriNet,
        petri_net_im: Marking,
        petri_net_fm: Marking,
        trace_net: PetriNet,
        trace_net_im: Marking,
        trace_net_fm: Marking,
    ) -> Dict[str, float]:
        """Extract token replay fitness features."""
        trace = self._trace_net_to_trace(trace_net)

        log = EventLog([trace])

        # Configure token replay parameters
        parameters = {
            token_replay.Parameters.SHOW_PROGRESS_BAR: False,
        }

        aligned_traces = executor.apply(
            log,
            petri_net,
            petri_net_im,
            petri_net_fm,
            variant=executor.Variants.TOKEN_REPLAY,
            parameters=parameters,
        )

        if not aligned_traces or len(aligned_traces) == 0:
            logging.warning("Token replay returned no aligned traces.")
            return {
                'token_replay_trace_is_fit': 0.0,
                'token_replay_trace_fitness': 0.0,
                'token_replay_missing_tokens': 0.0,
                'token_replay_consumed_tokens': 0.0,
                'token_replay_remaining_tokens': 0.0,
                'token_replay_produced_tokens': 0.0,
            }

        result = aligned_traces[0]

        return {
            'token_replay_trace_is_fit': float(
                result.get('trace_is_fit', False)
            ),
            'token_replay_trace_fitness': result.get('trace_fitness', 0.0),
            'token_replay_missing_tokens': float(
                result.get('missing_tokens', 0)
            ),
            'token_replay_consumed_tokens': float(
                result.get('consumed_tokens', 0)
            ),
            'token_replay_remaining_tokens': float(
                result.get('remaining_tokens', 0)
            ),
            'token_replay_produced_tokens': float(
                result.get('produced_tokens', 0)
            ),
        }

    def _trace_net_to_trace(self, trace_net: PetriNet) -> Trace:
        """
        Convert a trace Petri net (sequential net) to a PM4Py Trace object.

        Trace nets are linear Petri nets where transitions are connected
        sequentially: p0 -> t0 -> p1 -> t1 -> ... -> pn

        Args:
            trace_net: A sequential Petri net representing a trace

        Returns:
            A PM4Py Trace object with events for each transition in order
        """
        transitions = sorted(
            trace_net.transitions,
            key=lambda t: int(t.name.split('_')[-1]) if '_' in t.name else 0,
        )

        activity_key = xes_util.DEFAULT_NAME_KEY

        trace = Trace()
        for transition in transitions:
            if transition.label is not None:
                event = Event({activity_key: transition.label})
                trace.append(event)

        return trace
