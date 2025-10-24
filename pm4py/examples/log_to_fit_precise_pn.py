import pm4py
from pm4py.objects.conversion.trie import converter as trie_converter


def execute_script():
    log = pm4py.read_xes("../tests/input_data/running-example.xes")

    prefix_tree = pm4py.discover_prefix_tree(log)

    net, im, fm = trie_converter.apply(prefix_tree, variant=trie_converter.Variants.TO_PETRI_NET)

    pm4py.view_petri_net(net, im, fm, format="svg")


if __name__ == "__main__":
    execute_script()
