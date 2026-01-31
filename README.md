# Project Repository "Recommender System For Heuristics in A*-based Alignment Computation"

### Setup

Run setup.sh.
```
./setup.sh
```
This should set up a venv for you and correctly install pm4py as an editable module.

If .venv is not already active:
```
source .venv/bin/activate # MacOS/Linux
source .venv/bin/activate.[shell] # (e.g. activate.fish) if your shell requires and/or generates a special script.
source .venv/Scripts/activate # Windows
```

### Event-log data

There's a helper in `dataloaders` that allows bulk downloading of many real world event logs from `https://data.4tu.nl`.
To download the datasets listed in `dataloaders/sources.yaml` execute `python -m dataloaders.pull`.
For more information on data refer to `dataloaders/README.md`.


#### Caches

If you want to download prefilled alignment dataset caches:

It's recommended to setup rclone on your machine
```
./scripts/setup_rclone.sh
```

Go through the prompts and choose defaults except for:
1.	n → New remote
2.	Name: gdrive
3.  Option: 22 (drive)
4.	client_id: press Enter (leave blank)
5.	client_secret: press Enter (leave blank)
6.	Scope: choose "drive" → full read/write access
10.	Use auto config? → y
•	Browser opens → log in with Google account
•	Approve access
11.	Configure as shared drive? → n

Then you can download the cache contents
```
./download_gdrive_cache.sh
```

#### Important
```
./updload_gdrive_cache.sh
```
SYNCHS with the drive... it makes the remote look EXACTLY like your local folder.
If you expanded runs or only added new files to the folder this is fine. Just be aware that remote will mirror your local EXACTLY.


#### Running Experiments

The central script that runs the evaluation pipeline end-to-end is `scripts/evaluate_classifier_e2e.py`.
**IMPORTANT**:
If caches and tables are not populated, the script will begin running alignments on your machine to gather all data defined by the pipeline configuration in `configs/default.yaml`.

__Note__:
Even with populated caches the pipeline will check for data completeness/integrity. This step can take several minutes (between 20 - 40min depending on the amount of evaluation data) and consume quite a bit of RAM.
It is highly recommended to use precomputed tables for training where applicable (XGBoost + Baselines) with `--train-tables`.

**To run in-distribution testing:**
```
python -m scripts.evaluate_classifier_e2e --eval-mode iid --train-tables
```

**To run out-of-distribution testing:**
```
python -m scripts.evaluate_classifier_e2e --eval-mode ood --train-tables
```

If you have a CUDA capable device and a trained Transformer Model, ensure determinism by setting the appropriate CUBLAS environment variable:
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.evaluate_classifier_e2e <options>
```

##### DL-Model training
To train the GNN-Transformer model run:
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.model.train_transformer_model
````
If you don't care about determinism you can theoretically turn it off IF you have precalculated training batches.
We provide precalculated class balanced batches of the exact training split produced for the other models in `/cache`, however these are not automatically updated if data pipeline configurations change. In that case: delete the batch cache -> run the script as above and the batches will be reinstantiated as cache files for retraining/experimentation purposes.

__Note__:
DL-Model training is protected against CUDA out-of-memory errors, of which a hand full can occur during training even on higher end hardware, note however that this makes training outcome (if only slightly) hardware dependent.
A pretrained model 
