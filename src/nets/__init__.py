"""Problem-agnostic network components of the Attention Model (§3, App. A).

These modules are shared by every routing problem (TSP, CVRP, ...). Anything
problem-specific (input features, decoder context, mask, termination, cost)
lives under `src/problems/`.
"""
