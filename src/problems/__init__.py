"""Problem-specific pieces of the Attention Model.

Each routing problem (TSP, CVRP, ...) provides: an *embedder* (raw inputs ->
node embeddings), a *DecodeState* (context / mask / termination), a *decoder*
wrapper that owns any problem-specific learned parameters, an instance
*generator*, and a *cost* function. The shared encoder/decoder mechanics live in
`src/nets/`.
"""
