import pm4py
from pm4py.streaming.stream.live_event_stream import LiveEventStream
from pm4py.streaming.algo.conformance.declare import algorithm as declare_streaming_cc
import os


def execute_script():
    log = pm4py.read_xes(os.path.join("..", "tests", "input_data", "receipt.xes"))
    declare = pm4py.discover_declare(log, min_support_ratio=0.3, min_confidence_ratio=1.0)
    event_stream = pm4py.convert_to_event_stream(log)
    conf_obj = declare_streaming_cc.apply(declare)
    live_stream = LiveEventStream()
    live_stream.register(conf_obj)
    live_stream.start()
    for index, event in enumerate(event_stream):
        live_stream.append(event)
    live_stream.stop()
    diagn_df = conf_obj.get()
    print(diagn_df.keys())
    print(diagn_df["total_events_processed"])
    print(diagn_df["total_deviations"])


if __name__ == "__main__":
    execute_script()
