# Get Datasets

Download all data by running `python -m dataloaders.pull` from the ROOT of the project.
You can theoretically configure that script to download the data to anywhere but I recommend just leaving the defaults.

### IMPORTANT
If you already ran the first version and don't want to download everything again:
On distros/OS with perl based `rename`:
```
rename -n 's/^.*?_uuid_//' data/*_uuid_*
```

# Log Dataloaders

Essentially just a typed wrapper for one of two PM4PY import codepaths.


# Process Model Dataset

A process model dataset takes in any BaseEventLogDataset and induces/discovers process models from that event log.
You should provide a dictionary of discovery methods for that purpose.
dataloaders.net.DISCOVERY_METHODS.ALL is a default for all discovery functions contained in pm4py.
But you can also pass your own.
You should provide a dictionary of parameter lists, a "parameter grid" for the discovery algorithm.
All possible permutations of parameters that are valid for a given function will be run.
**The dataset thus contains all combinations of the product DISCOVERY_METHOD x PARAM_GRID**.
This means the dataset can get quite big.


# Runs Dataset

Runs Dataset is the pipeline for running alignement computations, recording their results and their profile.
A Run Dataset takes a Process Model Dataset as a parameter.

So overall we get BaseEventLogDataset -> ProcessModelDataset -> RunDataset.

An item in RunDataset should record everything we can record for a combination of ProcessModel x Trace (to be aligned) x Alignment method.

Right now this happens using CProfiler AND time.perf_counter() because the former has a hard time with exact timings in very short executions.

In runs.py `__main__` one can find an example of an end to end construction of a dataset starting from an xes file.

Initial construction can take quite a while. Process Discovery takes quite some time and running every process model against every trace with slight perturbations expands the total item set a lot.
One can restrict the number of traces taken from the original dataset by specifying a slice range: e.g. only try to align traces (10, 50).

After caching is complete, extracting a "labeled" dataset becomes quite simple by grouping on the ids of each dataset item.
Example for this is also in `__main__`.

Datasets are "incremental" meaning they store all previously generated data yet only access items that result from the current configuration.
This means that configurations can be "expanded" without recomputing already present items.


# Synthetic Process Models and Eventlogs

RunDataset's can also accept SyntheticProcessModelDataset's. When using SyntheticProcessModelDatasets with RunDataset one has to use a different Trace Sampler object (SyntheticTraceSampler) because Synthetic EventLogs
as well as Synthetic Process Models never materialize on disk. Both are resampled based on their unique and deterministic sampling configuration which in many cases is as fast or faster than read and deserializing from disk.
Essentially the flow from Eventlog to ProcessModel happens in reverse for Synthetic data. Which is why the API of RunDataset is not entirely congruent for this case.

TODOs:

    ✅ caching.
