import unittest
import os

import polars as pl
import pm4py

DIR_PATH = "input_data"


class TestPolarsStatistics(unittest.TestCase):

    def test_variants(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "receipt.csv")).lazy()
        from pm4py.statistics.variants.polars import get as variants_get
        variants_set = variants_get.get_variants_set(df)
        variants_count = variants_get.get_variants_count(df)

    def test_start_activities(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        from pm4py.statistics.start_activities.polars import get as sa_get
        start_activities = sa_get.get_start_activities(df)

    def test_end_activities(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        from pm4py.statistics.end_activities.polars import get as ea_get
        end_activities = ea_get.get_end_activities(df)

    def test_rework(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        from pm4py.statistics.rework.polars import get as rew_get
        rework_stats = rew_get.apply(df)

    def test_passed_time(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        df = df.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        from pm4py.statistics.passed_time.polars import algorithm as passed_time
        pre = passed_time.apply(df, "check ticket", variant=passed_time.Variants.PRE)
        post = passed_time.apply(df, "check ticket", variant=passed_time.Variants.POST)
        prepost = passed_time.apply(df, "check ticket", variant=passed_time.Variants.PREPOST)

    def test_service_time(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        df = df.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        from pm4py.statistics.service_time.polars import get as st_get
        service_time = st_get.apply(df)

    def test_traces_cycle_time(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        df = df.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        from pm4py.statistics.traces.cycle_time.polars import get as ct_get
        cycle_times = ct_get.apply(df)

    def test_traces_arrival(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        df = df.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        from pm4py.statistics.traces.generic.polars import case_arrival
        avg_arrival = case_arrival.get_case_arrival_avg(df)
        avg_dispersion = case_arrival.get_case_dispersion_avg(df)

    def test_traces_generic(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        df = df.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        from pm4py.statistics.traces.generic.polars import case_statistics
        durations = case_statistics.get_all_case_durations(df)
        variant_stats = case_statistics.get_variant_statistics(df)
        cases_desc = case_statistics.get_cases_description(df)
        median = case_statistics.get_median_case_duration(df)
        first_quartile = case_statistics.get_first_quartile_case_duration(df)
        variants_df = case_statistics.get_variants_df(df)
        kde = case_statistics.get_kde_caseduration(df)

    def test_overlap_cases(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        df = df.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        from pm4py.statistics.overlap.cases.polars import get as overlap_get
        overlap = overlap_get.apply(df)

    def test_overlap_interval_events(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        df = df.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        from pm4py.statistics.overlap.interval_events.polars import get as ie_get
        interval_events = ie_get.apply(df)

    def test_eventually_follows(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        df = df.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        from pm4py.statistics.eventually_follows.polars import get as efg_get
        efg = efg_get.apply(df, parameters={"keep_first_following": True})

    def test_concurrent_activities(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        df = df.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        from pm4py.statistics.concurrent_activities.polars import get as ca_get
        concurrent = ca_get.apply(df)

    def test_attributes_get1(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "running-example.csv")).lazy()
        df = df.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        from pm4py.statistics.attributes.polars import get as attr_get
        events_dist = attr_get.get_events_distribution(df)
        kde_date = attr_get.get_kde_date_attribute(df, "time:timestamp")
        values = attr_get.get_attribute_values(df, "concept:name")

    def test_attributes_get2(self):
        df = pl.read_csv(os.path.join(DIR_PATH, "roadtraffic100traces.csv")).lazy()
        df = df.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        from pm4py.statistics.attributes.polars import get as attr_get
        kde_numeric = attr_get.get_kde_numeric_attribute(df, "amount")


if __name__ == "__main__":
    unittest.main()
