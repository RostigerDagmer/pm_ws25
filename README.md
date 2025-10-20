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

---

### Read problem description [here](documents/Background%20Info.md)
### **Step-by-Step Plan to Solve the Problem**

The goal of our project is to build an intelligent system that can automatically select the best heuristic for a given alignment problem from a set of picked heuristics to minimize the total computation time. We will frame this as a **supervised machine learning problem**: given a process model and a trace, our system will predict which available heuristic will be the fastest.

---
To achieve this, we follow the structured plan:

**Step 1: Implement Heuristic Functions**


Our first task is to implement the set of candidate heuristics that our recommender will choose from. This will be done within the **PM4Py** Python library, see links: [GitHub](https://github.com/process-intelligence-solutions/pm4py), [Webpage](https://processintelligence.solutions/pm4py), [Conformance Source Code](https://pm4py-source.readthedocs.io/en/latest/_modules/pm4py/conformance.html). 
<br>We should implement multiple distinct approaches:
*   **A Simple Baseline Heuristic:** A fast-to-compute but less accurate heuristic that can serve as a baseline.
*   **An ILP-based Heuristic:** A highly accurate but computationally expensive heuristic.
*  **More?**

---

**Step 2: Generate Training Data**

To train a machine learning model, we need a large dataset of labeled examples. We will generate this dataset by running a series of experiments:
1.  For a large number of different process models and event traces:
2.  We will compute the optimal alignment for each model-trace pair **multiple times**, once with each of our implemented heuristics.
3.  For each run, we will precisely measure the total execution time.
4.  The heuristic that resulted in the shortest time will be designated as the "correct" golden label for that specific model-trace pair.
5.  We will then extract descriptive features from the model and trace (this is part of Step 3).
6.  Finally, we will store the feature vector and its corresponding label, creating one data point for our training set.

---

**Step 3: Learn a Recommendation Model**


With the labeled dataset from Step 2, we can train a machine learning classifier. This involves two key activities:
*   **Feature Engineering:** We need to define a set of features that can describe the complexity of an alignment problem. These features will be the input for our model. Examples include:
    *   *Model Features:* Number of places and transitions in the Petri net, measures of concurrency (parallel paths), measures of choice, etc.
    *   *Trace Features:* Length of the trace, number of unique activities, etc.
*   **Model Training:** We will use the engineered features and the "fastest heuristic" labels to train a classification model (e.g., a Decision Tree, Random Forest, or Gradient Boosting model). The model will learn the patterns that connect the features of a problem to the best-performing heuristic.

---

**Step 4: Experiments and Evaluation**
The final step is to  evaluate our solution to prove its effectiveness. We will split our generated data into a training set and a separate test set. On the test set, we will compare the performance of three scenarios:
1.  **Baseline A:** The average time taken when *always* using the simple heuristic.
2.  **Baseline B:** The average time taken when *always* using the ILP-based heuristic.
3.  **Our Recommender:** The average time taken when using the heuristic predicted by our trained model for each instance.

---

**Step 5: Documentation: Write a report**

