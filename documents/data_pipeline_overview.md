# Data Pipeline Overview

This document describes the complete data generation pipeline from raw event logs to individual training samples for the alignment heuristic selection model.

## Pipeline Architecture

The data pipeline consists of four main stages:

```
Event Logs → Process Models → Alignments → Training Samples
    ↓              ↓              ↓              ↓
 Loading      Discovery      Execution      Features + Labels
```

---

## Stage 1: Event Log Loading

**Location**: `dataloaders/`

### Components

- **`BaseEventLogDataset`**: Abstract base class for loading event logs
- **`XESEventLogDataset`**: Loads `.xes` format event logs
- **`CSVEventLogDataset`**: Loads `.csv` format event logs

### Purpose

Load event logs from files and provide them as PyTorch `Dataset` objects that yield PM4Py `Trace` objects.

### Data Structure

Each event log contains:
- **Traces**: Sequences of events representing process executions
- **Events**: Individual activities with attributes (timestamp, resource, etc.)
- **Metadata**: Case attributes, global log properties

### Example

```python
from dataloaders import XESEventLogDataset

# Load event log
dataset = XESEventLogDataset("data/some_log.xes")

# Access traces
trace = dataset[0]  # Returns a PM4Py Trace object
print(f"Trace length: {len(trace)}")
print(f"Activities: {[event['concept:name'] for event in trace]}")
```

### Output Format

- **Type**: `pm4py.objects.log.obj.Trace`
- **Content**: Sequence of events with attributes

---

## Stage 2: Process Model Generation

**Location**: `dataloaders/net.py`

### Components

- **`ProcessModelDataset`**: Wraps a `BaseEventLogDataset` and discovers Petri nets
- **`Sampler`**: Abstract class for trace subset selection strategies
  - `RandomSampler`: Random trace selection
  - `StratifiedSampler`: Stratified sampling by trace properties
  - `AllTracesSampler`: Uses all traces from the log

### Purpose

Generate process models (Petri nets) from event logs using various discovery algorithms and parameters.

### Discovery Configuration

The system generates all combinations of:

1. **Discovery Algorithms**:
   - Alpha Miner
   - Alpha+ Miner
   - Heuristics Miner
   - ILP Miner
   - Inductive Miner

2. **Parameter Grids**: Algorithm-specific parameters
   - Dependency threshold (Heuristics Miner)
   - Noise threshold (Inductive Miner)
   - etc.

3. **Sampling Strategies**: Different trace subsets for discovery

### Deduplication Pipeline

**Location**: `deduplication/`

To avoid redundant models, a three-stage deduplication process removes structurally similar Petri nets:

#### Stage 1: Transition Label Comparison
- Compares frequency distribution of transition labels
- Uses Bray-Curtis similarity metric
- Fast initial filter

#### Stage 2: Transition Edge Structure
- Compares edge structure between transitions
- Captures structural patterns (AND-splits, XOR-splits, etc.)
- Uses Bray-Curtis similarity on edge count distributions

#### Stage 3: Feature Vector Comparison
- Extracts full feature vectors from both nets
- Normalizes using z-scores
- Compares using Median Absolute Deviation (MAD)
- Most comprehensive but computationally expensive

Each stage has configurable similarity thresholds and can be enabled/disabled independently.

### Data Structure

Each process model consists of:
- **PetriNet**: Graph structure with places, transitions, arcs
- **InitialMarking**: Starting state (token distribution)
- **FinalMarking**: Accepting end state
- **TraceSubset**: Subset of traces used for discovery (with metadata)

### Example

```python
from dataloaders.net import ProcessModelDataset
from dataloaders.sampler import RandomSampler

# Create process model dataset
event_log_dataset = XESEventLogDataset("data/some_log.xes")
sampler = RandomSampler(sample_size=100)
model_dataset = ProcessModelDataset(
    event_log_dataset,
    sampler=sampler,
    enable_deduplication=True
)

# Access a process model
net, im, fm, trace_subset = model_dataset[0]
print(f"Transitions: {len(net.transitions)}")
print(f"Places: {len(net.places)}")
print(f"Discovered from {len(trace_subset)} traces")
```

### Output Format

- **Type**: Tuple of `(PetriNet, Marking, Marking, TraceSubset)`
- **Content**: Complete process model representation

### Configuration

Discovery parameters are specified in `configs/default.yaml`:

```yaml
process_model_generation:
  discovery_methods:
    - alpha
    - heuristics
    - inductive
  param_grids:
    heuristics:
      dependency_threshold: [0.5, 0.7, 0.9]
    inductive:
      noise_threshold: [0.0, 0.2, 0.5]
  deduplication:
    enabled: true
    thresholds:
      stage1_bray_curtis: 0.95
      stage2_bray_curtis: 0.90
      stage3_mad_threshold: 0.1
```

---

## Stage 3: Alignment Execution

**Location**: `dataloaders/runs.py`

### Components

- **`RunDataset`**: Wraps a `ProcessModelDataset` and executes alignments
- **`Aligner`**: Abstract interface for alignment algorithms
  - `DijkstraAligner`
  - `AStarILPAligner`
  - `AStarAligner`
  - `RequiredModelMoveAligner`
  - `RemainingActivitiesAligner`

### Purpose

For each (process model, trace) pair, execute alignments with all heuristic variants and measure execution time.

### Alignment Process

For each combination:

1. **Select Model**: Pick a process model from Stage 2
2. **Select Trace**: Pick a trace from the original event log
3. **Run Alignments**: Execute alignment with each heuristic
4. **Measure Time**: Record execution time for each heuristic
5. **Store Results**: Save timing data and alignment quality

### Noise Injection (Optional)

To test robustness, the system can inject noise into traces:

```python
from dataloaders.runs import inject_noise_trace

# Inject 10% noise (insertions, deletions, substitutions)
noisy_trace = inject_noise_trace(
    original_trace,
    noise_level=0.1,
    transition_labels=["A", "B", "C", "D"]
)
```

### Data Structure

Each run produces:

```python
{
    "combination_id": "uuid-1234",          # Unique ID for (model, trace) pair
    "model_hash": "hash-5678",              # Model identifier
    "trace_hash": "hash-9012",              # Trace identifier
    "aligner": "Dijkstra",                  # Heuristic name
    "time_total_mean": 0.152,               # Average execution time (seconds)
    "time_total_std": 0.008,                # Standard deviation
    "alignment_cost": 5,                    # Edit distance
    "alignment_fitness": 0.92,              # Quality metric
    "feature_vector": [15, 8, 23, ...]      # Combined features
}
```

### Output Format

Results are stored in CSV files:
- **Filename**: `<dataset_uuid>.runs.csv`
- **Location**: `cache/.runs/` (real datasets) or `cache/.runs_synthetic/` (synthetic)
- **Format**: One row per (model, trace, heuristic) combination

### Example

```python
from dataloaders.runs import RunDataset
from dataloaders.aligners import DijkstraAligner, AStarAligner

# Create run dataset
aligners = [DijkstraAligner(), AStarAligner()]
run_dataset = RunDataset(
    model_dataset,
    aligners=aligners,
    n_repetitions=3  # Run each alignment 3 times for stable timing
)

# Execute and collect results
for combination in run_dataset:
    print(f"Combination: {combination['combination_id']}")
    print(f"Best aligner: {combination['best_aligner']}")
    print(f"Fastest time: {combination['time_total_mean']:.3f}s")
```

---

## Stage 4: Feature Extraction & Label Creation

**Location**: `features/extractors.py`, `scripts/create_labels.py`

### Components

- **`ModelFeatureExtractor`**: Extracts features from Petri nets
- **`TraceFeatureExtractor`**: Extracts features from traces
- **`CompositeFeatureExtractor`**: Combines both + interaction features

### Model Features

Structural properties of the Petri net:

#### Basic Counts
- Number of transitions (total, visible, invisible)
- Number of places
- Number of arcs

#### Transition Label Statistics
- Unique transition labels
- Duplicate transitions (same label)
- Label frequency distribution

#### Split Patterns
- AND-splits: Number of places with multiple outgoing arcs
- XOR-splits: Number of transitions with multiple outgoing arcs
- Parallel paths
- Choice structures

#### Degree Statistics
Per transition/place type (visible/invisible, unique/duplicate):
- Mean, median, max degree (in-degree, out-degree)
- Degree variance

#### Spectral Features
Graph-based features using eigenvalue analysis:
- Algebraic connectivity (Fiedler value)
- Spectral gap
- Largest eigenvalue
- Graph diameter estimates

### Trace Features

Properties of the execution sequence:

- **Trace length**: Number of events
- **Unique activities**: Number of distinct activity types
- **Activity frequencies**: Distribution of activity occurrences
- **Repetition patterns**: Loops and repeated activities
- **Activity transitions**: First-order transition probabilities

### Interaction Features

Combined features capturing model-trace relationships:

- **Coverage**: Fraction of trace activities present in model
- **Activity overlap**: Jaccard similarity between trace and model activities
- **Complexity ratio**: Trace length / model size
- **Rare activities**: Activities in trace but infrequent in model

### Feature Vector

All features are combined into a single numpy array:

```python
from features import CompositeFeatureExtractor

extractor = CompositeFeatureExtractor()

# Extract features
feature_vector = extractor.extract(net, im, fm, trace)

# Get feature names
feature_names = extractor.feature_names

# Example output
print(f"Feature vector shape: {feature_vector.shape}")
# Feature vector shape: (87,)

print(f"Sample features:")
for name, value in zip(feature_names[:5], feature_vector[:5]):
    print(f"  {name}: {value}")
# Sample features:
#   model_n_transitions: 42
#   model_n_places: 38
#   model_n_arcs: 95
#   model_n_visible_transitions: 35
#   model_n_invisible_transitions: 7
```

### Label Creation

**Script**: `scripts/create_labels.py`

The label for each (model, trace) combination is the **fastest heuristic**:

1. Load runs.csv file
2. Group by `combination_id`
3. Find aligner with minimum `time_total_mean`
4. Assign as label

```python
import pandas as pd

# Load runs
df = pd.read_csv("cache/.runs/dataset.runs.csv")

# Find best heuristic for each combination
best_indices = df.groupby('combination_id')['time_total_mean'].idxmin()
labels = df.loc[best_indices, ['combination_id', 'aligner']]

# Label distribution
print(labels['aligner'].value_counts())
# Dijkstra             1523
# RemainingActivities   342
# A*-ILP                 89
# A*                     12
```

### Training Sample Format

Each training sample consists of:

```python
{
    "combination_id": "uuid-1234",
    "feature_vector": np.array([...]),  # Shape: (87,)
    "label": "Dijkstra",                 # Best heuristic
    "label_encoded": 0,                  # Integer encoding
    "time_best": 0.152,                  # Time of best heuristic
    "time_worst": 2.341,                 # Time of worst heuristic
    "speedup": 15.4                      # Speedup factor (worst/best)
}
```

---

## Synthetic Data Generation

**Location**: `dataloaders/synthetic.py`

For additional training data and controlled experiments, the system can generate synthetic process models and traces.

### Components

- **`SyntheticEventLogDataset`**: Generates synthetic event logs
- **Structured Net Generation**: Creates nets with specific patterns
  - Sequential structures
  - Parallel branches (AND-splits/joins)
  - Choice structures (XOR-splits/joins)
  - Loop structures

### Configuration

Synthetic generation is controlled via configuration:

```yaml
synthetic_data:
  num_nets: 100
  structure:
    sequential_probability: 0.3
    parallel_probability: 0.3
    choice_probability: 0.3
    loop_probability: 0.1
  size_distribution:
    transitions: {min: 10, max: 50}
    places: {min: 8, max: 45}
  noise:
    enabled: true
    levels: [0.0, 0.05, 0.1, 0.15]
```

### Noise Types

- **Insertion**: Add random activities
- **Deletion**: Remove activities from trace
- **Substitution**: Replace activities with others

---

## Data Storage Structure

```
pm_ws25/
├── data/                          # Raw event logs
│   ├── <uuid-1>/
│   │   └── dataset1.xes
│   └── <uuid-2>/
│       └── dataset2.csv
│
├── cache/
│   ├── .runs/                     # Real dataset results
│   │   ├── <uuid-1>.runs.csv
│   │   └── <uuid-2>.runs.csv
│   │
│   └── .runs_synthetic/           # Synthetic dataset results
│       └── synthetic.runs.csv
│
└── outputs/                       # Experiment outputs
    ├── models/                    # Trained classifiers
    ├── plots/                     # Visualizations
    └── evaluation/                # Performance metrics
```

---

## Running the Pipeline

### Step 1: Download Datasets

```bash
python -m dataloaders.pull
```

### Step 2: Generate Process Models & Run Alignments

```bash
python scripts/generate_dataset.py --config configs/default.yaml
```

This script:
1. Loads event logs
2. Discovers process models with deduplication
3. Runs alignments with all heuristics
4. Saves results to `cache/.runs/`

### Step 3: Create Labels

```bash
python scripts/create_labels.py
```

Processes runs.csv files and identifies the fastest heuristic for each combination.

### Step 4: Train Model

```bash
python experiments/train_classifier.py
```

Loads features and labels, trains a classifier to predict the best heuristic.

---

## Data Statistics

### Real Datasets

As of the current pipeline run:

- **Number of datasets**: 21 real event logs
- **Total process models**: ~120,000 (after deduplication)
- **Total (model, trace) combinations**: ~150,000
- **Total alignment runs**: ~750,000 (5 heuristics per combination)

### Feature Dimensions

- **Model features**: 65 dimensions
- **Trace features**: 8 dimensions
- **Interaction features**: 14 dimensions
- **Total features**: 87 dimensions

### Label Distribution (Aggregated)

Average across all datasets:

| Heuristic            | Percentage |
|----------------------|------------|
| Dijkstra             | 72.4%      |
| RemainingActivities  | 18.9%      |
| A*-ILP               | 6.2%       |
| A*                   | 1.8%       |
| RequiredModelMove    | 0.7%       |

---

## Key Design Decisions

### Why PyTorch Dataset Interface?

The pipeline uses PyTorch's `Dataset` abstraction for consistency and compatibility with ML frameworks, even though the initial stages don't involve neural networks.

### Why Deduplication?

Process discovery often generates structurally identical or very similar nets with different parameter settings. Deduplication:
- Reduces computational cost
- Avoids data leakage in train/test splits
- Improves model generalization

### Why Multiple Heuristics?

Different alignment heuristics excel in different scenarios:
- **Dijkstra**: Optimal but slow for large search spaces
- **A*-ILP**: Fast for highly structured models
- **RemainingActivities**: Good heuristic for most cases
- The goal is to automatically select the best one

### Why Feature Extraction?

Raw Petri nets and traces are graph structures, not suitable for ML models. Feature extraction converts them into fixed-size numerical vectors that capture relevant structural properties.

---

## Future Enhancements

Potential improvements to the pipeline:

1. **Incremental Processing**: Process new datasets without recomputing everything
2. **Distributed Execution**: Parallelize alignment runs across multiple machines
3. **Online Learning**: Update model as new data arrives
4. **Active Learning**: Intelligently select which combinations to run
5. **Deep Learning**: End-to-end learning from graph structures
6. **Multi-Objective Optimization**: Balance speed and alignment quality

---

## References

- PM4Py documentation: https://pm4py.fit.fraunhofer.de/
- Process mining book: "Process Mining: Data Science in Action" by Wil van der Aalst
- Alignment algorithms: "Replaying history on process models for conformance checking and performance analysis" (2012)
