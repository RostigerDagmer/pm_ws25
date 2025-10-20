# Project Background: Heuristic Recommendation for Alignments

### **1. Understanding the Problem Domain: Process Mining and Conformance Checking**

Our project is situated within the field of **process mining**, a discipline that bridges data science and business process management. The main goal of process mining is to discover, monitor, and improve real-world processes by analyzing the digital footprints they leave behind in information systems. These footprints are collected in **event logs**, which record what activity was performed, for which case (e.g., an order or a patient), and when it happened.

Process mining is typically divided into three main categories:
1.  **Process Discovery:** Automatically creating a process model from an event log.
2.  **Conformance Checking:** Comparing an event log (reality) against an existing process model (the desired or prescribed process) to identify and analyze deviations.
3.  **Process Enhancement:** Using insights from the event log to improve or extend an existing process model.

Our project focuses specifically on **conformance checking**.
<br>The fundamental challenge in this area is to move beyond a simple "yes/no" answer regarding compliance. A meaningful conformance analysis must provide detailed diagnostics, explaining precisely where, how, and why the observed behavior deviates from the modeled behavior. It is this diagnostic power that makes the technique of alignments, the core of our project, a state-of-the-art method in the field.

### **1.2 The Primary Artifacts: Event Logs and Process Models**

At the heart of any process mining task are two key artifacts: the **event log**, representing reality, and the **process model**, representing the intended or designed process.

An event log is the starting point for any process mining analysis. Formally, it can be defined as a multiset of traces, where each trace is a finite sequence of events. Each event corresponds to a single activity that occurred during a specific process execution. The minimal information required for each event is:   

1. **Case ID**: A unique identifier that groups all events belonging to a single execution of the process (e.g., an order ID, a patient ID).

2. **Activity**: A description of the task that was performed (e.g., 'Record Goods Receipt', 'Examine Thoroughly').

3. **Timestamp**: The time at which the event occurred, which allows for ordering the events within a case to form a trace.

Event logs often contain a wealth of additional data, such as the resource who performed the activity, costs, or other case-specific attributes, which can be used for more advanced analyses. 

### 1.3 Process Models: Petri Nets

While many formalisms exist to describe process models, such as Business Process Model and Notation (BPMN) or Process Trees, your project specifically requires a working knowledge of Petri nets. Petri nets are a formal, graphical language for modeling the behavior of distributed systems and are particularly well-suited for process mining due to their ability to represent concurrency.


A Petri net consists of four basic components :   

* **Places:** Represented by circles, places denote conditions or states within the process. For example, a place could represent the state "waiting for approval."
* **Transitions:** Represented by rectangles, transitions denote events or activities that can occur, causing a change in state. A transition might be labeled with an activity name like 'Approve Request'.
* **Arcs:** Directed edges that connect places to transitions and transitions to places, defining the flow and dependencies of the process.
* **Tokens:** Represented by black dots inside places, tokens define the current state of the process model. This state is also known as the marking of the net.

The behavior of a Petri net is defined by the "firing rule": a transition is enabled if all of its input places (places with an arc leading to the transition) contain at least one token. When an enabled transition fires, it consumes one token from each input place and produces one token in each of its output places (places to which the transition has an outgoing arc). A sequence of firings constitutes an execution path through the model.

Petri nets can model the three fundamental control-flow patterns found in business processes :   

*  **Sequence:** Activity B can only happen after Activity A. This is modeled by having the output place of the transition for A be the input place for the transition for B.

*  **Choice (XOR-split/join):** After an activity, a choice is made between two or more mutually exclusive paths. This is modeled by a place with outgoing arcs to multiple transitions. The firing of one transition consumes the token, disabling the others.

*  **Parallelism (AND-split/join):** Two or more activities can be executed in any order or at the same time. This is modeled by a transition with multiple output places, placing a token in each and thus enabling multiple downstream paths simultaneously.

Understanding these components and their dynamics is a prerequisite for comprehending how the state space of a process model is explored during the alignment computation.

---

### **2. The Core Technique: Understanding Alignments**

Read: [Wil-Aalst 1](https://dl.acm.org/doi/fullHtml/10.1145/2240236.2240257)
[Wil-Aalst 2](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1045?saml_referrer)

Alignments are the central technique for conformance checking in modern process mining. They provide a formal and intuitive way to relate the observed behavior in an event log to the modeled behavior in a process model, offering a detailed diagnosis of any discrepancies.

### **2.1 What is an Alignment?**

Formally, an alignment is a pairwise matching between a trace from an event log and a valid firing sequence in the process model that starts at the initial marking and ends in a final marking. The objective is to find the "closest model run" that corresponds to the observed trace.

If a trace perfectly conforms to the model, the alignment is a straightforward one-to-one mapping of each event in the trace to a corresponding transition firing in the model. However, the real power of alignments becomes evident when dealing with non-conforming behavior. If a trace deviates, a perfect mapping is impossible. In such cases, the alignment must introduce specific "moves" to explain the discrepancies between what was observed (the log) and what was possible (the model). The result is a sequence of pairs, where the top element represents a step in the log trace and the bottom element represents a step in the model's execution sequence.

### 2.2 Alignment Moves

Every step in an alignment falls into one of three categories, which together form the "language" used to describe conformance and deviation. These moves are summarized below:


1.  **Synchronous Move:** An event in the log perfectly matches an allowed activity in the model. This represents **conforming behavior**. <br>**Represented as a pair (activity, activity)**
2.  **Log Move:** An event occurred in the log, but that activity was not permitted by the model at that point. This represents a deviation, such as an **unexpected activity**.  <br>**Represented as (activity, >>)**
3.  **Model Move:** An activity was expected or required by the model, but it did not occur in the log. This represents a different kind of deviation, such as a **skipped activity**. <br> **Represented as (>>, activity)**

By combining these moves, any trace from the log can be perfectly related to a valid path in the model, with all deviations explicitly highlighted.

However, for any given deviation, there are often many possible alignments. To find the most plausible explanation, we use a **cost function** that assigns a penalty to each deviating move (log moves and model moves). Synchronous moves typically have a cost of **zero**. The **optimal alignment** is the one with the minimum total cost, representing the simplest and most likely explanation for the observed behavior.

#### Note:
It is crucial to recognize that the resulting optimal alignment, and therefore the diagnosis of the process deviation, is entirely dependent on the chosen cost function. Manually defining these costs based on domain knowledge is common but can be subjective and error-prone. An illogical cost function can lead to an optimal alignment that provides a mathematically correct but contextually nonsensical explanation for a deviation.

---


### **3. The Computational Challenge: The A\* Search Algorithm**

The computation of an optimal alignment is a computationally intensive task. It is framed as a shortest-path problem on a massive state-space graph that represents all possible combinations of log events and model states. <br> The size of this graph can grow exponentially, especially with complex business processes, making a simple search computationally infeasible.
<br>This is solved using an efficient, informed search algorithm known as A* ("A-star"). "Heuristic Recommendation for Alignments," is directly concerned with optimizing this search process.

### **3.1 Alignment as a Shortest-Path Problem**

Finding the optimal alignment between a log trace and a process model is equivalent to finding the shortest path in a large state-space graph. This graph is conceptually constructed as follows:

* **The Graph (Synchronous Product Net):** The search space is formally defined by the **synchronous product** of the state space of the process model (the reachability graph of the Petri net) and the sequence of events in the log trace. An illustration of such a product net is on the introduction slides, **Figure 7.3**.


* **Nodes (States):** Each node, or state, in this search graph is a tuple of the form (marking, trace_position). This represents being in a specific state of the Petri net (defined by its current marking) while having processed the log trace up to a certain event. The initial state is (initial_marking, 0).


* **Edges (Arcs):** An edge between two states represents a single legal alignment move (synchronous, log, or model) that transitions from one state to the next.


* **Edge Weights:** The weight of each edge is the cost of the corresponding alignment move, as defined by the cost function.

The goal is to find the path with the minimum total weight from the initial state to a final state, where the final state is any state of the form (final_marking, length_of_trace).


### 3.2 The A\* Algorithm

[A/* interactive Intro](https://www.redblobgames.com/pathfinding/a-star/introduction.html)

To solve this problem efficiently, process mining tools use the **A\* ("A-star") search algorithm**. A\* is an intelligent search algorithm that finds the shortest path much faster than exhaustive methods by using a heuristic to guide its search. At each step, A\* evaluates which path to explore next using the function:

$f(n) = g(n) + h(n)$ 

Where:
*   $g(n)$ is the **actual cost** of the path from the start to the current state $n$. This is a known value.
*   $h(n)$ is the **heuristic function**, which provides an *estimated* cost of the cheapest path from the current state $n$ to the goal. This is an "educated guess."

The algorithm prioritizes exploring paths with the lowest $f(n)$ value.

---

### **4. The Role of Heuristics**

The performance and correctness of the A\* algorithm depend entirely on the quality of the heuristic function, $h(n)$. To guarantee that A\* finds the true optimal alignment, the heuristic must be **admissible**, meaning it **never overestimates** the actual remaining cost.

This leads to a fundamental trade-off that is the central problem of our project:

*   **Simple, Fast Heuristics:** These are very quick to calculate but provide a poor estimate of the remaining cost. This forces the A\* algorithm to explore a very large number of states, making the overall alignment computation slow.
*   **Complex, Accurate Heuristics:** These provide a much better estimate, allowing A\* to prune the search space and explore far fewer states. However, calculating the heuristic itself at each step is computationally expensive. A state-of-the-art method for this uses **Integer Linear Programming (ILP)**, which is highly accurate but can be slow to compute.

The total time to find an alignment is **the sum of the time spent computing heuristics plus the time spent exploring states**. Neither heuristic is universally better; the best choice depends on the specific characteristics of the process model and the event trace being analyzed.

---
