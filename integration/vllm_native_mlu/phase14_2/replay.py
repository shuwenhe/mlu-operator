"""Run the frozen TP=2 acceptance driver with the Vendor lifecycle executor."""
from functools import partial


if __name__ == "__main__":
    from ..phase14_1 import engine
    from .executor import LifecycleExecutor

    engine.LLM = partial(engine.LLM, distributed_executor_backend=LifecycleExecutor)
    engine.main()
