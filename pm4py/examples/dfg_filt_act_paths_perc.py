import pm4py
from examples import examples_conf
import importlib.util


def execute_script():
    log = pm4py.read_xes("../tests/input_data/receipt.xes")
    dfg, start_act, end_act = pm4py.discover_dfg(log)
    # keep the specified amount of activities
    dfg, start_act, end_act = pm4py.filter_dfg_activities_percentage(dfg, start_act, end_act, percentage=0.3)
    # keep the specified amount of paths
    dfg, start_act, end_act = pm4py.filter_dfg_paths_percentage(dfg, start_act, end_act, percentage=0.3)

    if importlib.util.find_spec("graphviz"):
        # view the DFG
        pm4py.view_dfg(dfg, start_act, end_act, format="svg")


if __name__ == "__main__":
    execute_script()
