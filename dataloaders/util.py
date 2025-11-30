from pathlib import Path
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.obj import EventLog, Trace, EventStream
from dataloaders.csv_log import CSVEventLogDataset
from dataloaders.xes_log import XESEventLogDataset
from collections.abc import Sequence


def _normalize_log_input(subset) -> pd.DataFrame | EventLog | Trace:
    """
    Normalize 'subset' to a pm4py-compatible pandas DataFrame.
    Supports:
      - pandas.DataFrame
      - pm4py EventLog / EventStream
      - any Sequence (e.g., list, tuple, TraceSubset) of pm4py Traces
      - any Sequence of event dicts
    """
    # 1) Already a DataFrame
    if isinstance(subset, pd.DataFrame):
        return subset

    # 2) pm4py log types
    if isinstance(subset, (EventLog, EventStream)):
        return log_converter.apply(
            subset, variant=log_converter.Variants.TO_DATA_FRAME
        )

    # 3) Generic sequences (includes your TraceSubset), excluding text
    if isinstance(subset, Sequence) and not isinstance(subset, (str, bytes)):
        seq = list(subset)

        if len(seq) == 0:
            raise ValueError(
                "Empty subset: cannot discover a model from zero events/traces."
            )

        # 3a) Sequence of pm4py Traces -> wrap into EventLog then to DataFrame
        if all(isinstance(t, Trace) for t in seq):
            evlog = EventLog(seq)
            return log_converter.apply(
                evlog, variant=log_converter.Variants.TO_DATA_FRAME
            )

        # 3b) Sequence of event dicts -> directly to DataFrame
        if all(isinstance(e, dict) for e in seq):
            return pd.DataFrame(seq)

        # Mixed or unsupported element type
        raise TypeError(
            "Unsupported sequence element types for subset: expected all pm4py Traces or all event dicts; "
            f"got examples like {type(seq[0])!r}."
        )

    # 4) Anything else -> unsupported
    raise TypeError(f"Unsupported subset type: {type(subset)}")


CONSTRUCTION_PARAMS = {
    # HEADER:
    # CustomerID;AgeCategory;Gender;Office_U;Office_W;ContactDate;ContactTimeStart;ContactTimeEnd;QuestionThemeID;QuestionSubthemeID;QuestionTopicID;QuestionTheme;QuestionSubtheme;QuestionTopic;QuestionTheme_EN;QuestionSubtheme_EN;QuestionTopic_EN
    "2b02709f-9a84-4538-a76a-eb002eacf8d1": {
        "rtype": "csv",
        "case_id_col": "CustomerID",
        "timestamp_col": "ContactTimeStart",
        "activity_col": "QuestionTopic_EN",
        "sep": ";",
    },
    # HEADER:
    # CI Name (aff);CI Type (aff);CI Subtype (aff);Service Component WBS (aff);Incident ID;Status;Impact;Urgency;Priority;Category;KM number;Alert Status;# Reassignments;Open Time;Reopen Time;Resolved Time;Close Time;Handle Time (Hours);Closure Code;# Related Interactions;Related Interaction;# Related Incidents;# Related Changes;Related Change;CI Name (CBy);CI Type (CBy);CI Subtype (CBy);ServiceComp WBS (CBy)
    "3cfa2260-f5c5-44be-afe1-b70d35288d6d": {
        "rtype": "csv",
        "case_id_col": "CI Name (aff)",
        "timestamp_col": "Open Time",
        "activity_col": "ServiceComp WBS (CBy)",
        "sep": ";",
    },
    # HEADER:
    # CI Name (aff);CI Type (aff);CI Subtype (aff);Service Comp WBS (aff);Interaction ID;Status;Impact;Urgency;Priority;Category;KM number;Open Time (First Touch);Close Time;Closure Code;First Call Resolution;Handle Time (secs);Related Incident
    "3d5ae0ce-198c-4b5c-b0f9-60d3035d07bf": {
        "rtype": "csv",
        "case_id_col": "CI Name (aff)",
        "timestamp_col": "Open Time (First Touch)",
        "activity_col": "CI Type (aff)",
        "sep": ";",
    },
    # HEADER:
    # SessionID;IPID;TIMESTAMP;VHOST;URL_FILE;PAGE_NAME;REF_URL_category;page_load_error;page_action_detail;tip;service_detail;xps_info;page_action_detail_EN;service_detail_EN;tip_EN
    "9b99a146-51b5-48df-aa70-288a76c82ec4": {
        "rtype": "csv",
        "case_id_col": "SessionID",
        "timestamp_col": "TIMESTAMP",
        "activity_col": "page_action_detail_EN",
        "sep": ";",
    },
    # HEADER:
    # CustomerID;AgeCategory;Gender;Office_U;Office_W;SessionID;IPID;TIMESTAMP;VHOST;URL_FILE;PAGE_NAME;REF_URL_category;page_load_error;page_action_detail;tip;service_detail;xps_info;page_action_detail_EN;service_detail_EN;tip_EN
    "01345ac4-7d1d-426e-92b8-24933a079412": {
        "rtype": "csv",
        "case_id_col": "CustomerID",
        "timestamp_col": "TIMESTAMP",
        "activity_col": "page_action_detail_EN",
        "sep": ";",
    },
    # HEADER:
    # Incident ID;DateStamp;IncidentActivity_Number;IncidentActivity_Type;Assignment Group;KM number;Interaction ID
    "86977bac-f874-49cf-8337-80f26bf5d2ef": {
        "rtype": "csv",
        "case_id_col": "Incident ID",
        "timestamp_col": "DateStamp",
        "activity_col": "IncidentActivity_Type",
        "sep": ";",
    },
    # HEADER:
    # CustomerID;AgeCategory;Gender;Office_U;Office_W;EventDateTime;EventType;HandlingChannelID
    "c3f3ba2d-e81e-4274-87c7-882fa1dbab0d": {
        "rtype": "csv",
        "case_id_col": "CustomerID",
        "timestamp_col": "EventDateTime",
        "activity_col": "Office_U",
        "sep": ";",
    },
    # HEADER:
    # CustomerID;AgeCategory;Gender;Office_U;Office_W;ComplaintDossierID;ComplaintID;ContactDate;ContactChannelID;ComplaintThemeID;ComplaintSubthemeID;ComplaintTopicID;ComplaintTheme;ComplaintSubtheme;ComplaintTopic;ComplaintTheme_EN;ComplaintSubtheme_EN;ComplaintTopic_EN
    "e30ba0c8-0039-4835-a493-6e3aa2301d3f": {
        "rtype": "csv",
        "case_id_col": "CustomerID",
        "timestamp_col": "ContactDate",
        "activity_col": "ComplaintTopic_EN",
        "sep": ";",
    },
}

DEFAULT_PARAMS_CSV = {
    "case_id_col": "case:concept:name",
    "activity_col": "concept:name",
    "timestamp_col": "time:timestamp",
    "sep": ",",
}

DEFAULT_PARAMS_XES = {"attribute": "concept:name"}


def build_dataset(path: str):
    ext = Path(path).suffix.lower()

    if ext == ".xes":
        return XESEventLogDataset(path)

    if ext == ".csv":
        dataset_id = Path(path).parent.name  # or another identifier
        if dataset_id not in CONSTRUCTION_PARAMS:
            raise KeyError(
                f"No CSV event log constructor params for '{dataset_id}'"
            )
        return CSVEventLogDataset(path, **CONSTRUCTION_PARAMS[dataset_id])

    raise ValueError(f"Unsupported log format: {ext}")
