import pm4py
import os
import traceback


def execute_script():
    log = pm4py.read_xes(os.path.join("..", "tests", "input_data", "receipt.xes"))

    # can be anything here :))
    process_tree = pm4py.discover_process_tree_inductive(log)

    # can be anything here too :)))
    powl = pm4py.discover_powl(pm4py.filter_variants_top_k(log, 20))

    # computes the footprints-based behavioral similarity between a process tree
    # and a POWL model
    behavioral_similarity = pm4py.behavioral_similarity(powl, process_tree)
    print("behavioral similarity", behavioral_similarity)

    # computes the structural similarity between a POWL and a process tree
    structural_similarity = pm4py.structural_similarity(process_tree, powl)
    print("structural similarity", structural_similarity)

    # computes the label-sets-similarity between a process tree and a POWL
    label_sets_similarity = pm4py.label_sets_similarity(powl, process_tree)
    print("label sets similarity", label_sets_similarity)

    try:
        # computes the embeddings-based similarity between a process tree and a POWL
        embeddings_similarity = pm4py.embeddings_similarity(powl, process_tree)
        print("embeddings similarity", embeddings_similarity)

        # embeddings are non-deterministic, so even comparing a model with itself,
        # it would probably give a value below 1
        embeddings_similarity2 = pm4py.embeddings_similarity(process_tree, process_tree)
        print("embeddings similarity", embeddings_similarity2)
    except:
        traceback.print_exc()


if __name__ == "__main__":
    execute_script()
