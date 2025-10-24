import pm4py
from pm4py.visualization.variants_duration import visualizer


def execute_script():
    log = pm4py.read_xes("../tests/input_data/receipt.xes")

    # visualize the variants durations aligning on the start
    gviz = visualizer.apply(log, parameters={"format": "svg", "alignment_criteria": "start"})
    visualizer.view(gviz)

    # visualize the variants durations aligning on the end
    gviz = visualizer.apply(log, parameters={"format": "svg", "alignment_criteria": "end"})
    visualizer.view(gviz)

    # visualize the variants aligning on (the first occurrence of) a given activity
    gviz = visualizer.apply(log, parameters={"format": "svg", "alignment_criteria": "T02 Check confirmation of receipt", "max_variants": 10})
    visualizer.view(gviz)


if __name__ == "__main__":
    execute_script()
