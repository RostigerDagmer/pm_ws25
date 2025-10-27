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


CONSTRUCTION_PARAMS = {
    # HEADER:
    # CustomerID;AgeCategory;Gender;Office_U;Office_W;ContactDate;ContactTimeStart;ContactTimeEnd;QuestionThemeID;QuestionSubthemeID;QuestionTopicID;QuestionTheme;QuestionSubtheme;QuestionTopic;QuestionTheme_EN;QuestionSubtheme_EN;QuestionTopic_EN
    "dx.doi.org_10.4121_uuid_2b02709f-9a84-4538-a76a-eb002eacf8d1": {
        "rtype": "csv",
        "case_id_col": "CustomerID",
        "timestamp_col": "ContactTimeStart",
        "activity_col": "QuestionTopic_EN",
        "sep": ";",
    },
    # HEADER:
    # CI Name (aff);CI Type (aff);CI Subtype (aff);Service Component WBS (aff);Incident ID;Status;Impact;Urgency;Priority;Category;KM number;Alert Status;# Reassignments;Open Time;Reopen Time;Resolved Time;Close Time;Handle Time (Hours);Closure Code;# Related Interactions;Related Interaction;# Related Incidents;# Related Changes;Related Change;CI Name (CBy);CI Type (CBy);CI Subtype (CBy);ServiceComp WBS (CBy)
    "dx.doi.org_10.4121_uuid_3cfa2260-f5c5-44be-afe1-b70d35288d6d": {
        "rtype": "csv",
        "case_id_col": "Incident ID",
        "timestamp_col": "Open Time",
        "activity_col": "CI Name (aff)",
        "sep": ";",
    },
    # HEADER:
    # CI Name (aff);CI Type (aff);CI Subtype (aff);Service Comp WBS (aff);Interaction ID;Status;Impact;Urgency;Priority;Category;KM number;Open Time (First Touch);Close Time;Closure Code;First Call Resolution;Handle Time (secs);Related Incident
    "dx.doi.org_10.4121_uuid_3d5ae0ce-198c-4b5c-b0f9-60d3035d07bf": {
        "rtype": "csv",
        "case_id_col": "Interaction ID",
        "timestamp_col": "Open Time (First Touch)",
        "activity_col": "Category",  # ?
        "sep": ";",
    },
    # HEADER:
    # SessionID;IPID;TIMESTAMP;VHOST;URL_FILE;PAGE_NAME;REF_URL_category;page_load_error;page_action_detail;tip;service_detail;xps_info;page_action_detail_EN;service_detail_EN;tip_EN
    "dx.doi.org_10.4121_uuid_9b99a146-51b5-48df-aa70-288a76c82ec4": {
        "rtype": "csv",
        "case_id_col": "SessionID",
        "timestamp_col": "TIMESTAMP",
        "activity_col": "page_action_detail_EN",
        "sep": ";",
    },
    # HEADER:
    # CustomerID;AgeCategory;Gender;Office_U;Office_W;SessionID;IPID;TIMESTAMP;VHOST;URL_FILE;PAGE_NAME;REF_URL_category;page_load_error;page_action_detail;tip;service_detail;xps_info;page_action_detail_EN;service_detail_EN;tip_EN
    "dx.doi.org_10.4121_uuid_01345ac4-7d1d-426e-92b8-24933a079412": {
        "rtype": "csv",
        "case_id_col": "CustomerID",
        "timestamp_col": "TIMESTAMP",
        "activity_col": "page_action_detail_EN",
        "sep": ";",
    },
    # HEADER:
    # Incident ID;DateStamp;IncidentActivity_Number;IncidentActivity_Type;Assignment Group;KM number;Interaction ID
    "dx.doi.org_10.4121_uuid_86977bac-f874-49cf-8337-80f26bf5d2ef": {
        "rtype": "csv",
        "case_id_col": "Incident ID",
        "timestamp_col": "DateStamp",
        "activity_col": "IncidentActivity_Type",
        "sep": ";",
    },
    # HEADER:
    # CustomerID;AgeCategory;Gender;Office_U;Office_W;EventDateTime;EventType;HandlingChannelID
    "dx.doi.org_10.4121_uuid_c3f3ba2d-e81e-4274-87c7-882fa1dbab0d": {
        "rtype": "csv",
        "case_id_col": "CustomerID",
        "timestamp_col": "EventDateTime",
        "activity_col": "Office_U",
        "sep": ";",
    },
    # HEADER:
    # CustomerID;AgeCategory;Gender;Office_U;Office_W;ComplaintDossierID;ComplaintID;ContactDate;ContactChannelID;ComplaintThemeID;ComplaintSubthemeID;ComplaintTopicID;ComplaintTheme;ComplaintSubtheme;ComplaintTopic;ComplaintTheme_EN;ComplaintSubtheme_EN;ComplaintTopic_EN
    "dx.doi.org_10.4121_uuid_e30ba0c8-0039-4835-a493-6e3aa2301d3f": {
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
