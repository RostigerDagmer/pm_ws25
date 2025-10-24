import pm4py
from pm4py.statistics.process_cube.pandas import algorithm as process_cube_builder
import traceback


def execute_script():
    dataframe = pm4py.read_xes("../tests/input_data/receipt.xes")

    # builds a complete feature table, including the case ID as first column
    enriched_df = pm4py.extract_outcome_enriched_dataframe(dataframe)
    feature_table = pm4py.extract_features_dataframe(enriched_df, include_case_id=True)

    # prints the columns of the feature table
    for col in feature_table.columns:
        #if col.startswith("@@"):
        print(col)

    try:
        # builds a first process cube
        # let's put as rows the different activities, as columns the different resources, and a value aggregated
        # the total time of the case
        cube_df, cell_case_dict = process_cube_builder.apply(feature_table, x_col="concept:name", y_col="org:resource", agg_col="@@sojourn_time")

        print(cube_df)
        #print(cell_case_dict)

        # in this case, each cell would aggregate on the cases containing at least one event of the given activity
        # and at least one event of the given resource (not necessarily the same)

        # using the results in `cell_case_dict`, we can filter the dataframe on a given cell.
        # for example,
        filt_df1 = dataframe[dataframe["case:concept:name"].isin(cell_case_dict[('concept:name_T17 Check report Y to stop indication_y', 'org:resource_Resource01_x')])]
        print(filt_df1)
    except:
        traceback.print_exc()
        #input()

    try:
        # let's build another process cube!
        # as rows, I want the presence (or not) of the activity 'T06 Determine necessity of stop advice'
        # as columns, I want the channel used to submit the request

        cube_df2, cell_case_dict2 = process_cube_builder.apply(feature_table, x_col="concept:name_T06 Determine necessity of stop advice_y", y_col="case:channel", agg_col="@@sojourn_time")

        print(cube_df2)
        #print(cell_case_dict2)

        # using the results in `cell_case_dict`, we can filter the dataframe on a given cell.
        # for example, let's filter on the cases reaching the 'Desk' channel and containing zero occurrences
        # of the activity 'T06 Determine necessity of stop advice'

        cell = [x for x in cell_case_dict2 if x[1] == 'case:channel_Desk_x' and x[0].left < 0 < x[0].right][0]
        filt_df2 = dataframe[dataframe["case:concept:name"].isin(cell_case_dict2[cell])]
        #print(filt_df2)
    except:
        traceback.print_exc()
        #input()

    try:
        # let's try to build another process cube
        # as rows, we divide cases based on their arrival rate (time between start of the current case and the start of the previously started case)
        # as columns, we divide cases based on their finish rate (time between end of the nextly terminated case and the end of the current case)

        cube_df3, cell_case_dict3 = process_cube_builder.apply(feature_table, x_col="@@arrival_rate", y_col="@@finish_rate", agg_col="@@sojourn_time")

        print(cube_df3)
        #print(cell_case_dict3)

        # now, let's focus on the cases which distance from the previously started case and from the nextly terminated
        # case is lower than one day
        cell = [x for x in cell_case_dict3 if x[0].left < 86400 < x[0].right and x[1].left < 86400 < x[1].right][0]
        filt_df3 = dataframe[dataframe["case:concept:name"].isin(cell_case_dict3[cell])]
        print(filt_df3)
    except:
        traceback.print_exc()
        #input()

    try:
        # let's try to build another process cube
        # as rows, we divide cases based on their arrival rate (time between start of the current case and the start of the previously started case)
        # as columns, we divide cases based on their finish rate (time between end of the nextly terminated case and the end of the current case)

        cube_df3, cell_case_dict3 = process_cube_builder.apply(feature_table, x_col="@@arrival_rate", y_col="@@finish_rate", agg_col="@@sojourn_time", parameters={
            "x_bins": [0, 200000, 500000, 800000, 1000000],
            "y_bins": [0, 300000, 600000, 900000, 1000000]
        })

        print(cube_df3)
        #print(cell_case_dict3)

        # now, let's focus on the cases which distance from the previously started case and from the nextly terminated
        # case is lower than one day
        cell = [x for x in cell_case_dict3 if x[0].left < 86400 < x[0].right and x[1].left < 86400 < x[1].right][0]
        filt_df3 = dataframe[dataframe["case:concept:name"].isin(cell_case_dict3[cell])]
        print(filt_df3)
    except:
        traceback.print_exc()
        #input()


if __name__ == "__main__":
    execute_script()
