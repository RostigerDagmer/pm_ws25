# %%
import pandas as pd
import pm4py

# from pm4py/notebooks/4_conformance_checking.ipynb
df = pm4py.format_dataframe(
    pd.read_csv('../pm4py/notebooks/data/running_example.csv', sep=';'),
    case_id='case_id',
    activity_key='activity',
    timestamp_key='timestamp',
)
pn, im, fm = pm4py.discover_petri_net_inductive(df)
pm4py.view_petri_net(
    pn, im, fm
)  # should display a graphviz plot of the petrinet.
# %%
