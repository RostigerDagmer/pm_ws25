import os
import unittest

import polars as pl
import pm4py


DIR_PATH = "input_data"


class TestPolarsStatisticsSimplified(unittest.TestCase):
    def _load_running_example(self) -> pl.LazyFrame:
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        return df.with_columns(
            pl.col("time:timestamp").str.strptime(
                pl.Datetime, "%Y-%m-%d %H:%M:%S%z"
            ),
            pl.col("case:concept:name").cast(pl.Utf8),
            pl.col("concept:name").cast(pl.Utf8),
        )

    def test_variants(self):
        df = (
            pl.read_csv(os.path.join(DIR_PATH, "receipt.csv"))
            .lazy()
            .with_columns(
                pl.col("time:timestamp").str.strptime(
                    pl.Datetime, "%Y-%m-%d %H:%M:%S.%f%z"
                ),
                pl.col("case:concept:name").cast(pl.Utf8),
                pl.col("concept:name").cast(pl.Utf8),
            )
        )
        variants = pm4py.get_variants(df)
        self.assertTrue(len(variants) > 0)

    def test_start_activities(self):
        df = self._load_running_example()
        start_activities = pm4py.get_start_activities(df)
        self.assertTrue(start_activities)

    def test_end_activities(self):
        df = self._load_running_example()
        end_activities = pm4py.get_end_activities(df)
        self.assertTrue(end_activities)

    def test_event_attribute_values(self):
        df = self._load_running_example()
        attribute_values = pm4py.get_event_attribute_values(df, "concept:name")
        self.assertIn("register request", attribute_values)

    def test_trace_attribute_values(self):
        df = self._load_running_example()
        trace_values = pm4py.get_trace_attribute_values(df, "creator")
        self.assertTrue(trace_values)

    def test_case_arrival_average(self):
        df = self._load_running_example()
        avg_arrival = pm4py.get_case_arrival_average(df)
        self.assertGreaterEqual(avg_arrival, 0)

    def test_rework_cases_per_activity(self):
        df = self._load_running_example()
        rework_stats = pm4py.get_rework_cases_per_activity(df)
        self.assertTrue(rework_stats)

    def test_case_overlap(self):
        df = self._load_running_example()
        overlap = pm4py.get_case_overlap(df)
        self.assertTrue(len(overlap) > 0)

    def test_cycle_time(self):
        df = self._load_running_example()
        cycle_time = pm4py.get_cycle_time(df)
        self.assertGreaterEqual(cycle_time, 0)

    def test_service_time(self):
        df = self._load_running_example()
        service_time = pm4py.get_service_time(
            df,
            aggregation_measure="mean",
            start_timestamp_key="time:timestamp",
        )
        self.assertTrue(service_time)

    def test_all_case_durations(self):
        df = self._load_running_example()
        durations = pm4py.get_all_case_durations(df)
        self.assertTrue(durations)

    def test_case_duration(self):
        df = self._load_running_example()
        case_ids = (
            df.select(pl.col("case:concept:name").unique())
            .collect()
            .get_column("case:concept:name")
        )
        first_case = case_ids[0]
        duration = pm4py.get_case_duration(df, first_case)
        self.assertGreaterEqual(duration, 0)


if __name__ == "__main__":
    unittest.main()
