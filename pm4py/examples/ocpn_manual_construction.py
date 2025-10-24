import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as token_based_replay


def get_tbr_statistics(log, net, im, fm):
    tbr_parameters = {"enable_pltr_fitness": True, "show_progress_bar": False}

    replayed_traces, place_fitness_per_trace, transition_fitness_per_trace, notexisting_activities_in_model = token_based_replay.apply(log, net, im, fm, parameters=tbr_parameters)
    place_diagnostics = {place: {"m": 0, "r": 0, "c": 0, "p": 0}  for place in place_fitness_per_trace}
    trans_count = {trans: 0 for trans in net.transitions}
    # computes the missing, remaining, consumed, and produced tokens
    # per place.
    for place, res in place_fitness_per_trace.items():
        place_diagnostics[place]["m"] += res["m"]
        place_diagnostics[place]["r"] += res["r"]
        place_diagnostics[place]["c"] += res["c"]
        place_diagnostics[place]["p"] += res["p"]
    # counts the number of times a transition has been fired during the
    # replay.
    for trace in replayed_traces:
        for trans in trace["activated_transitions"]:
            trans_count[trans] += 1
    return (place_diagnostics, trans_count)


def execute_script():
    log = pm4py.read_xes("../tests/input_data/running-example.xes")

    net, im, fm = pm4py.discover_petri_net_inductive(log)

    ocpn = {}
    ocpn["petri_nets"] = {}
    ocpn["double_arcs_on_activity"] = {}
    ocpn["tbr_results"] = {}
    ocpn["activities"] = set()

    for ot in {"ciao1", "ciao2"}:
        for trans in net.transitions:
            if trans.label is not None:
                ocpn["activities"].add(trans.label)

        ocpn["petri_nets"][ot] = [net, im, fm]

        ocpn["double_arcs_on_activity"][ot] = {}
        for x in net.transitions:
            ocpn["double_arcs_on_activity"][ot][x.label] = True if ot == "ciao2" and x.label is not None else False

        ocpn["tbr_results"][ot] = pm4py.conformance_diagnostics_token_based_replay(log, net, im, fm)

        ocpn["tbr_results"][ot] = get_tbr_statistics(log, net, im, fm)

    pm4py.view_ocpn(ocpn, format="svg")


if __name__ == "__main__":
    execute_script()
