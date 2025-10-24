import pm4py, os
from pm4py.algo.concept_drift import algorithm as concept_drift_detection


def execute_script():
    log = pm4py.read_xes(os.path.join("..", "tests", "input_data", "receipt.xes"))

    # Bose's concept drift detection is solely based on the control-flow perspective
    returned_sublogs, change_timestamps, p_values = concept_drift_detection.apply(log, parameters={"max_no_change_points": 3})

    # print the timestamps and the p-values of the detected change points
    print(change_timestamps)
    print(p_values)

    # print each one of the returned sub-logs (that is a Pandas dataframe)
    for sl in returned_sublogs:
        print(sl)
        dfg, sa, ea = pm4py.discover_dfg(sl)
        pm4py.view_dfg(dfg, sa, ea, format="svg")


if __name__ == "__main__":
    execute_script()
