# Phase 14.1 — Native TP=2 worker architecture

Use a distinct architecture key for the TP=2 implementation. A packaged Vendor general-plugin entry point runs before model resolution in each worker. The official model loader calls the model's weight-loading hook. The legacy model and its pre-existing worktree changes remain outside this implementation.

The worker owns the eight-head BF16 KV views and the engine owns their block IDs. The model validates scheduler spans, local page bounds and rank metadata agreement before dispatching any numerical collectives. It uses the frozen native head-wise MLA and the Phase 13.3.3 ordered O continuation. The frozen replicated decoder helpers handle non-MLA computation.
