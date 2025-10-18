# Project Repository for trace alignment research on Product Petrinets

### Repo structure

Experiments (scripts) go into /experiments.
Notebooks go into /notebooks.
Documents go into /documents. Like the report (/report) and management stuff (anything that goes into a markdown file but not into a notebook).
pm4py is a fork -> submodule so we can integrate right here if we want to.

### Setup

Initialize submodules if you did not clone --recursive.
```
git submodule update --init --recursive
```

Run setup.sh.
```
./setup.sh
```
This should set up a venv for you.

If .venv is not already active:
```
source .venv/bin/activate # MacOS/Linux
source .venv/bin/activate.[shell] # (e.g. activate.fish) if your shell requires and/or generates a special script.
source .venv/Scripts/activate # Windows
```
