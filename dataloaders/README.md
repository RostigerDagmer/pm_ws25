# Get Datasets

Download all data by running `python -m dataloaders.pull` from the ROOT of the project.
You can theoretically configure that script to download the data to anywhere but I recommend just leaving the defaults.

## Dataloaders

Because loading the data using pm4py is trivial and I presume we're going to use torch for the ML part I already created Dataloaders for CSV and XES.
To be as generic as possible w.r.t. feature definition in the dataloader the constructor takes both a **vocab_fn** and a **feature_fn** Callable argument that you have to provide.

##### vocab_fn

Has to map trace attributes to indices or dicts of key -> value for each attribute. Think of it like a tokenizer over the vocabulary of the attributes of the trace.
There are two variants already in the file of the BaseClass.
One builds a nested dict the other builds a flat dict.

##### feature_fn

Is a builder for the actual feature function that has to map a trace to a feature vector.
You provide the dataset with a function taking in the vocab you have built or is produced by one of the defaults.
You return a function that takes a trace and returns a tensor of features.

An example is also in the file of the base class.


## Process Model Dataset

A process model dataset takes in any BaseEventLogDataset and induces/discovers process models from that event log.
You should provide a dictionary of discovery methods for that purpose.
dataloaders.net.DISCOVERY_METHODS.ALL is a default for all discovery functions contained in pm4py.
But you can also pass your own.
You should provide a dictionary of parameter lists, a "parameter grid" for the discovery algorithm.
All possible permutations of parameters that are valid for a given function will be run.
**The dataset thus contains all combinations of product DISCOVERY_METHOD x PARAM_GRID**.
This means the dataset can get quite big... especially if you use the PARAM_GRID.EXTENSIVE default.

TODOs:

    [] caching.

