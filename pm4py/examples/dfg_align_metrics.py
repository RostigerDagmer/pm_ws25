import pm4py
from pm4py.algo.conformance.alignments.dfg.variants import classic as dfg_alignments


def execute_script():
    """
    Executes the DFG-based alignment conformance checking process.

    The process involves the following steps:
      1. Read an event log from an XES file.
      2. Filter the event log to include only the top 5 variants.
      3. Discover the Directly-Follows Graph (DFG) along with its start and end activities.
      4. Compute the DFG-based alignments against the original event log, which projects the alignment results onto the DFG.
         - The conformance-annotated DFG shows the number of SYNC (synchronous) and MM (model moves)
           executions for each edge.
         - The conformance-annotated activity dictionary shows the number of SYNC, MM, and LM (log moves)
           executions for each activity.
      5. Print the resulting conformance-annotated DFG and activity dictionary.
    """
    # Read the event log from the XES file
    log = pm4py.read_xes("../tests/input_data/receipt.xes")

    # Filter the event log to the top 5 variants and discover the DFG, along with start (sa) and end (ea) activities
    dfg, sa, ea = pm4py.discover_dfg(pm4py.filter_variants_top_k(log, 5))

    # Compute the DFG-based alignments and project the results on the DFG.
    # The function returns two items:
    #   - conformance_dfg: a dictionary where each edge is annotated with the count of SYNC and MM moves.
    #   - activities_conformance: a dictionary where each activity is annotated with the count of SYNC, MM, and LM moves.
    conformance_dfg, activities_conformance = dfg_alignments.project_alignments_on_dfg(log, dfg, sa, ea)

    # Output the conformance-annotated DFG and activities to the console
    print("Conformance DFG:")
    print(conformance_dfg)
    print("\nConformance Activities:")
    print(activities_conformance)


if __name__ == "__main__":
    # Execute the script if the module is run as the main program.
    execute_script()
