
import os
import time
from pm4py.algo.conformance.alignments.petri_net import algorithm as ali
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as petri_importer
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net
from pm4py.vis import view_petri_net, view_alignments
from pm4py.objects.conversion.log import converter as log_converter

# Make sure your python path is set to the project root

if __name__ == '__main__':
    # Load the log and the petri net
    log_path = os.path.join("pm4py", "tests", "input_data", "running-example.xes")
    log = xes_importer.apply(log_path)
    pnml_path = os.path.join("pm4py", "tests", "input_data", "running-example.pnml")
    net, marking, fmarking = petri_importer.apply(pnml_path)

    # Visualize petri net
    view_petri_net(net,marking,fmarking)

    # Construct trace net for the first trace and view it
    trace = log._list[0]
    trace_net, trace_im, trace_fm = construct_trace_net(trace)
    view_petri_net(trace_net, trace_im, trace_fm)

    # Print the log
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    print("\n", df.to_string(), "\n")

    timings = {}
    alignments = {}

    # Compute alignments using different variants and measure time
    for variant in ali.Variants:
        
        start = time.time()
        # Asynchronous variant leads to error; ChatGPT: this is a known issue in pm4py
        alignment = ali.apply(log._list[0], net, marking, fmarking,
                               variant=variant,
                               parameters={ali.Parameters.SYNCHRONOUS:True,ali.Parameters.EXPONENT:1.1})
        end = time.time()
        
        timings[variant] = end - start
        alignments[variant] = alignment

    # assert all heuristics produce the same alignment
    assert all(alignments[variant]["alignment"] == alignments[ali.Variants.VERSION_DISCOUNTED_A_STAR]["alignment"] for variant in ali.Variants)
    
    print("\nAlignment timings:")
    for variant in ali.Variants:
        print(f"Variant: {variant}, Time: {timings[variant]}s")

    # View the alignment
    view_alignments([log._list[0]], [list(alignments.values())[0]])
