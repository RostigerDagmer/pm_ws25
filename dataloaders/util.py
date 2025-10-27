from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.obj import EventLog, Trace
import pandas as pd


def _normalize_log_input(subset):
    """
    Normalize subset to a pm4py-compatible log object (preferably a pandas DataFrame).
    Supports: list of traces, EventLog, pandas DataFrame.
    """
    if isinstance(subset, pd.DataFrame):
        return subset

    # Already pm4py EventLog
    if isinstance(subset, EventLog):
        return log_converter.apply(
            subset, variant=log_converter.Variants.TO_DATA_FRAME
        )

    # List of traces
    if isinstance(subset, list):
        if all(isinstance(t, Trace) for t in subset):
            # Convert list of traces -> EventLog
            event_log = EventLog(subset)
            return log_converter.apply(
                event_log, variant=log_converter.Variants.TO_DATA_FRAME
            )

        # Already a list of dicts/events
        if all(isinstance(e, dict) for e in subset):
            return pd.DataFrame(subset)

        raise TypeError(
            f"Unsupported subset type: list elements must be traces or event dicts, got {type(subset[0])}"
        )

    raise TypeError(f"Unsupported subset type: {type(subset)}")
