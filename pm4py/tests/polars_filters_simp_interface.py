import os
import unittest

import polars as pl
import pm4py


DIR_PATH = "input_data"


class TestPolarsFilteringSimplified(unittest.TestCase):
    DO_PRINT = False
    ENABLE_LAZY = True

    def _print(self, dataframe):
        if self.DO_PRINT:
            print(dataframe)

    def _lazy(self, dataframe):
        if self.ENABLE_LAZY:
            return dataframe.lazy()
        return dataframe

    def _collect(self, dataframe):
        if self.ENABLE_LAZY:
            return dataframe.collect()
        return dataframe

    def _load_running_example(self):
        df = self._lazy(pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")))
        return df.with_columns(
            pl.col("time:timestamp").str.strptime(
                pl.Datetime, "%Y-%m-%d %H:%M:%S%z"
            ),
            pl.col("case:concept:name").cast(pl.Utf8),
            pl.col("concept:name").cast(pl.Utf8),
        )

    def _load_road_traffic(self):
        df = self._lazy(pl.read_csv(os.path.join(DIR_PATH, "roadtraffic100traces.csv")))
        return df.with_columns(
            pl.col("time:timestamp").str.strptime(
                pl.Datetime, "%Y-%m-%d %H:%M:%S%z"
            ),
            pl.col("case:concept:name").cast(pl.Utf8),
            pl.col("concept:name").cast(pl.Utf8),
        )

    def test_attributes_textual(self):
        df = self._load_running_example()
        filtered = pm4py.filter_event_attribute_values(
            df, "concept:name", ["pay compensation"], level="case"
        )
        self._print(self._collect(filtered))

        filtered_events = pm4py.filter_event_attribute_values(
            df, "concept:name", ["pay compensation"], level="event"
        )
        self._print(self._collect(filtered_events))

    def test_attributes_numeric(self):
        df = self._load_road_traffic()
        numeric_filtered = pm4py.filter_event_attribute_values(
            df, "amount", [5, 100], retain=True, level="event"
        )
        self._print(self._collect(numeric_filtered))

    def test_activity_split(self):
        df = self._load_running_example()
        split = pm4py.filter_between(df, "reinitiate request", "decide")
        self._print(self._collect(split))

    def test_between(self):
        df = self._load_running_example()
        between = pm4py.filter_between(df, "check ticket", "decide")
        self._print(self._collect(between))

    def test_cases(self):
        df = self._load_running_example()
        filtered_size = pm4py.filter_case_size(df, 2, 5)
        self._print(self._collect(filtered_size))

        filtered_perf = pm4py.filter_case_performance(df, 0, 86400 * 10)
        self._print(self._collect(filtered_perf))

    def test_end_activities(self):
        df = self._load_running_example()
        end_filtered = pm4py.filter_end_activities(df, ["pay compensation"], retain=True)
        self._print(self._collect(end_filtered))

    def test_start_activities(self):
        df = self._load_running_example()
        start_filtered = pm4py.filter_start_activities(df, ["register request"], retain=True)
        self._print(self._collect(start_filtered))

    def test_prefixes(self):
        df = self._load_running_example()
        prefixes = pm4py.filter_prefixes(
            df, "check ticket", strict=False, first_or_last="first"
        )
        self._print(self._collect(prefixes))

    def test_suffixes(self):
        df = self._load_running_example()
        suffixes = pm4py.filter_suffixes(
            df, "pay compensation", strict=False, first_or_last="last"
        )
        self._print(self._collect(suffixes))

    def test_eventually_follows(self):
        df = self._load_running_example()
        eventually = pm4py.filter_eventually_follows_relation(
            df, [("register request", "pay compensation")]
        )
        self._print(self._collect(eventually))

    def test_paths(self):
        df = self._load_running_example()
        paths = pm4py.filter_directly_follows_relation(
            df, [("examine casually", "check ticket")]
        )
        self._print(self._collect(paths))

    def test_rework(self):
        df = self._load_running_example()
        rework = pm4py.filter_activities_rework(df, "decide", 2)
        self._print(self._collect(rework))

    def test_time_range(self):
        df = self._load_running_example()
        lower = df.select(pl.col("time:timestamp").min()).collect()[0, 0]
        upper = df.select(pl.col("time:timestamp").max()).collect()[0, 0]
        time_filtered = pm4py.filter_time_range(df, lower, upper, mode="events")
        self._print(self._collect(time_filtered))


if __name__ == "__main__":
    unittest.main()
