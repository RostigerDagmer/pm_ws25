import pm4py
import os


def execute_script():
    log = pm4py.read_xes(os.path.join("..", "tests", "input_data", "running-example.xes"))

    # discovers a process moedl from the log
    process_tree = pm4py.discover_process_tree_inductive(log)
    # gets the label from the process models
    labels = pm4py.get_activity_labels(process_tree)
    # modify the labels
    labels = {x: x+"ADD" for x in labels}

    # replace the labels on the process model
    process_tree_mod = pm4py.replace_activity_labels(labels, process_tree)
    # view the modified process tree
    pm4py.view_process_tree(process_tree_mod, format="svg")

    # now, let's replace the labels back, with labels projection from another process model
    process_tree_mod2 = pm4py.map_labels_from_second_model(process_tree_mod, process_tree)
    # show the process tree modified
    pm4py.view_process_tree(process_tree_mod2, format="svg")


if __name__ == "__main__":
    execute_script()
