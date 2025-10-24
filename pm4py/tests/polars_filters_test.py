import unittest
import polars as pl


class TestPolarsFiltering(unittest.TestCase):
    """Test cases for Polars filtering functionality"""

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

    def test_attributes_textual(self):
        from pm4py.algo.filtering.polars.attributes import attributes_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        df2 = self._collect(attributes_filter.apply(dataframe, ["pay compensation"]))
        self._print(df2)
        df3 = self._collect(attributes_filter.apply(dataframe, ["aaaaaa"]))
        self._print(df3)
        df2 = self._collect(attributes_filter.apply_events(dataframe, ["pay compensation"]))
        self._print(df2)
        df3 = self._collect(attributes_filter.apply_events(dataframe, ["aaaaaa"]))
        self._print(df3)

    def test_attributes_numeric(self):
        from pm4py.algo.filtering.polars.attributes import attributes_filter
        dataframe = self._lazy(pl.read_csv("input_data/roadtraffic100traces.csv"))
        self._print(self._collect(dataframe))
        df2 = self._collect(
            attributes_filter.apply_numeric(dataframe, 5, 100, parameters={"pm4py:param:attribute_key": "amount"}))
        self._print(df2)
        df3 = self._collect(
            attributes_filter.apply_numeric(dataframe, 100, 1000, parameters={"pm4py:param:attribute_key": "amount"}))
        self._print(df3)
        df4 = self._collect(attributes_filter.apply_numeric(dataframe, 10000, 100000,
                                                            parameters={"pm4py:param:attribute_key": "amount"}))
        self._print(df4)

        df2 = self._collect(attributes_filter.apply_numeric_events(dataframe, 5, 100,
                                                                   parameters={"pm4py:param:attribute_key": "amount"}))
        self._print(df2)
        df3 = self._collect(attributes_filter.apply_numeric_events(dataframe, 100, 1000,
                                                                   parameters={"pm4py:param:attribute_key": "amount"}))
        self._print(df3)
        df4 = self._collect(attributes_filter.apply_numeric_events(dataframe, 10000, 100000,
                                                                   parameters={"pm4py:param:attribute_key": "amount"}))
        self._print(df4)

    def test_activity_split(self):
        from pm4py.algo.filtering.polars.activity_split import activity_split_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        df2 = self._collect(
            activity_split_filter.apply(dataframe, "reinitiate request", parameters={"cut_mode": "this"}))
        self._print(df2)
        df3 = self._collect(
            activity_split_filter.apply(dataframe, "reinitiate request", parameters={"cut_mode": "next"}))
        self._print(df3)
        df4 = self._collect(activity_split_filter.apply(dataframe, "ciao1", parameters={"cut_mode": "next"}))
        self._print(df4)

    def test_attr_value_repetition(self):
        from pm4py.algo.filtering.polars.attr_value_repetition import filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        df2 = self._collect(filter.apply(dataframe, "decide"))
        self._print(df2)
        df3 = self._collect(filter.apply(dataframe, "reinitiate request"))
        self._print(df3)
        df4 = self._collect(filter.apply(dataframe, "ciao1"))
        self._print(df4)

    def test_between(self):
        from pm4py.algo.filtering.polars.between import between_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        df2 = self._collect(between_filter.apply(dataframe, "check ticket", "decide"))
        self._print(df2)
        df3 = self._collect(between_filter.apply(dataframe, "ciao1", "ciao2"))
        self._print(df3)

    def test_cases(self):
        from pm4py.algo.filtering.polars.cases import case_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        dataframe = dataframe.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        df2 = self._collect(case_filter.filter_on_case_size(dataframe, "case:concept:name", 5, 10))
        self._print(df2)
        df2 = self._collect(case_filter.filter_on_case_size(dataframe, "case:concept:name", 50, 100))
        self._print(df2)
        df3 = self._collect(
            case_filter.filter_on_case_performance(dataframe, "case:concept:name", "time:timestamp", 86400, 86400 * 10))
        self._print(df3)
        df3 = self._collect(
            case_filter.filter_on_case_performance(dataframe, "case:concept:name", "time:timestamp", 86400 * 100,
                                                   86400 * 1000))
        self._print(df3)
        df4 = self._collect(case_filter.filter_on_ncases(dataframe, "case:concept:name", 3))
        self._print(df4)

    def test_consecutive_act_case_grouping(self):
        from pm4py.algo.filtering.polars.consecutive_act_case_grouping import consecutive_act_case_grouping_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        dataframe = dataframe.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        dataframe = pl.concat([dataframe, dataframe])
        dataframe = dataframe.sort(["case:concept:name", "time:timestamp"])
        self._print(self._collect(dataframe))
        df2 = self._collect(consecutive_act_case_grouping_filter.apply(dataframe, parameters={"filter_type": "first"}))
        self._print(df2)
        df3 = self._collect(consecutive_act_case_grouping_filter.apply(dataframe, parameters={"filter_type": "last"}))
        self._print(df3)

    def test_end_activities(self):
        from pm4py.algo.filtering.polars.end_activities import end_activities_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        self._print(end_activities_filter.get_end_activities(dataframe))
        df2 = self._collect(end_activities_filter.apply(dataframe, ["pay compensation"]))
        self._print(df2)
        df3 = self._collect(end_activities_filter.apply(dataframe, ["pay compensation", "reject request"]))
        self._print(df3)
        df4 = self._collect(end_activities_filter.apply(dataframe, ["aaaaaa"]))
        self._print(df4)

    def test_ends_with(self):
        from pm4py.algo.filtering.polars.ends_with import ends_with_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        self._print(ends_with_filter.get_variants_df(dataframe))
        df2 = self._collect(ends_with_filter.apply(dataframe, [("decide", "pay compensation")]))
        self._print(df2)
        df3 = self._collect(
            ends_with_filter.apply(dataframe, [("decide", "pay compensation"), ("decide", "reject request")]))
        self._print(df3)
        df4 = self._collect(ends_with_filter.apply(dataframe, [("ciao1", "ciao2")]))
        self._print(df4)

    def test_ltl(self):
        from pm4py.algo.filtering.polars.ltl import ltl_checker
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))

        df2 = self._collect(ltl_checker.four_eyes_principle(dataframe, ["check ticket", "register request"]))
        self._print(df2)
        df3 = self._collect(ltl_checker.eventually_follows(dataframe, ["register request", "pay compensation"]))
        self._print(df3)
        df4 = self._collect(ltl_checker.eventually_follows(dataframe, ["register request", "reject request"]))
        self._print(df4)
        df5 = self._collect(ltl_checker.eventually_follows(dataframe, ["register request", "raaaaa"]))
        self._print(df5)

    def test_paths(self):
        from pm4py.algo.filtering.polars.paths import paths_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        dataframe = dataframe.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        df2 = self._collect(paths_filter.apply(dataframe, [("examine casually", "check ticket")]))
        self._print(df2)
        df3 = self._collect(
            paths_filter.apply(dataframe, [("examine casually", "check ticket"), ("check ticket", "examine casually")]))
        self._print(df3)
        df4 = self._collect(paths_filter.apply_performance(dataframe, ("examine casually", "check ticket"),
                                                           parameters={"min_performance": 3600,
                                                                       "max_performance": 86400}))
        self._print(df4)
        df5 = self._collect(paths_filter.apply_performance(dataframe, ("examine casually", "check ticket"),
                                                           parameters={"min_performance": 3600,
                                                                       "max_performance": 3601}))
        self._print(df5)

    def test_prefixes(self):
        from pm4py.algo.filtering.polars.prefixes import prefix_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        df2 = self._collect(prefix_filter.apply(dataframe, [("register request", "check ticket")]))
        self._print(df2)
        df3 = self._collect(prefix_filter.apply(dataframe, [("register request", "check ticket"),
                                                            ("register request", "examine casually")]))
        self._print(df3)
        df4 = self._collect(prefix_filter.apply(dataframe, [("ciao1", "ciao2")]))
        self._print(df4)

    def test_rework(self):
        from pm4py.algo.filtering.polars.rework import rework_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        df2 = self._collect(rework_filter.apply(dataframe, parameters={"min_occurrences": 3}))
        self._print(df2)
        df3 = self._collect(rework_filter.apply(dataframe, parameters={"min_occurrences": 100}))
        self._print(df3)

    def test_start_activities(self):
        from pm4py.algo.filtering.polars.start_activities import start_activities_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        df2 = self._collect(start_activities_filter.apply(dataframe, ["register request"]))
        self._print(df2)
        df3 = self._collect(start_activities_filter.apply(dataframe, ["reject request"]))
        self._print(df3)

    def test_starts_with(self):
        from pm4py.algo.filtering.polars.starts_with import starts_with_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        df2 = self._collect(starts_with_filter.apply(dataframe, [("register request", "check ticket")]))
        self._print(df2)
        df3 = self._collect(starts_with_filter.apply(dataframe, [("register request", "check ticket"),
                                                                 ("register request", "examine casually")]))
        self._print(df3)
        df4 = self._collect(starts_with_filter.apply(dataframe, [("ciao1", "ciao2")]))
        self._print(df4)

    def test_suffixes(self):
        from pm4py.algo.filtering.polars.suffixes import suffix_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        df2 = self._collect(suffix_filter.apply(dataframe, [("decide", "pay compensation")]))
        self._print(df2)
        df3 = self._collect(
            suffix_filter.apply(dataframe, [("decide", "pay compensation"), ("decide", "reject request")]))
        self._print(df3)
        df4 = self._collect(suffix_filter.apply(dataframe, [("ciao1", "ciao2")]))
        self._print(df4)

    def test_timestamp_filter(self):
        from pm4py.algo.filtering.polars.timestamp import timestamp_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        dataframe = dataframe.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        df2 = self._collect(
            timestamp_filter.filter_traces_contained(dataframe, "2011-01-01 00:00:00", "2011-12-31 00:00:00"))
        self._print(df2)
        df2 = self._collect(
            timestamp_filter.filter_traces_contained(dataframe, "2009-01-01 00:00:00", "2009-12-31 00:00:00"))
        self._print(df2)
        df3 = self._collect(
            timestamp_filter.filter_traces_intersecting(dataframe, "2011-01-01 00:00:00", "2011-12-31 00:00:00"))
        self._print(df3)
        df3 = self._collect(
            timestamp_filter.filter_traces_intersecting(dataframe, "2009-01-01 00:00:00", "2009-12-31 00:00:00"))
        self._print(df3)
        df4 = self._collect(timestamp_filter.apply_events(dataframe, "2011-01-01 00:00:00", "2011-12-31 00:00:00"))
        self._print(df4)
        df4 = self._collect(timestamp_filter.apply_events(dataframe, "2009-01-01 00:00:00", "2009-12-31 00:00:00"))
        self._print(df4)

    def test_timestamp_case_grouping(self):
        from pm4py.algo.filtering.polars.timestamp_case_grouping import timestamp_case_grouping_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        dataframe = dataframe.with_columns(
            pl.col("time:timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z")
        )
        dataframe = pl.concat([dataframe, dataframe])
        # dataframe = dataframe.sort(["case:concept:name", "time:timestamp"])
        self._print(self._collect(dataframe))
        df2 = self._collect(timestamp_case_grouping_filter.apply(dataframe))
        self._print(df2)

    def test_traces(self):
        from pm4py.algo.filtering.polars.traces import trace_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        df2 = self._collect(trace_filter.apply(dataframe, [["register request", "...", "pay compensation"]]))
        self._print(df2)

    def test_variants(self):
        from pm4py.algo.filtering.polars.variants import variants_filter
        dataframe = self._lazy(pl.read_csv("input_data/running-example.csv"))
        df2 = self._collect(variants_filter.apply(dataframe, [
            ("register request", "examine casually", "check ticket", "decide", "pay compensation")]))
        self._print(df2)
        df3 = self._collect(variants_filter.apply(dataframe, [("ciao")]))
        self._print(df3)


if __name__ == "__main__":
    unittest.main()
