
**Synthetic is easy and valuable, but it’s also where people accidentally create degenerate distributions.**

Generators to implement:
1. Random WF-nets with structure knobs:
   Parameters: branching factor, concurrency rate, loop probability, %silent transitions, depth. Ensure single source/sink (WF-net); reject unsound ones (simple checks).
2. Block-structured nets:
   random compositions of sequence / XOR / AND / loop. Gives controllable structure.
3. Trace simulation: simulate the model to get “conforming” logs; then inject noise:
   - Insertions/deletions/substitutions (with label confusions).
   - Burst loops (repeat a sub-sequence).
   - Reordering within concurrency windows.

---
#### TODOs

- Ask for reference datasets
- Implement a simple generator that can reproduce at least the reference distributions.
- Scale the generator...