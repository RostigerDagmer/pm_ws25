experiments/simulate/structured_net.py
experiments/simulate/model.py

---
### StructuredNet

Is essentially just a tuple of (PN, im, fm) with a label/name attached.
**PN** is a Petrinet representing the semantics of the workflow.
**im** the _initial_ marking(s)
**fm** the _final_ marking(s)

##### DSL/Algebra

Defines algebraic operators congruent to the operators of the Process Tree "Language".

	'>>': Sequence  => A >> B reads as: "A preceeds B"
	'&':  Parallel  => A & B  reads as "A and B are concurrent"
	'^':  Choice    => A ^ B  reads as "Either A or B runs"
	'@':  loop      => A @ B  reads as "A loops with exit B"

In the block structured Workflownet representation they represent the analogous Process Tree operators **exactly**.

#### Tensor Representation

The default Simulator in pm4py is extremely slow.
A net with 150+ places takes on the order of minutes to emit an EventLog on the order of thousands of elements.

See [[Simulator]] for more on how to improve this.

The **StructureNet** wrapper therefore exposes a `to_tensor()` function which converts it to a suitable tensor representation in a shape the [[Simulator]] expects.