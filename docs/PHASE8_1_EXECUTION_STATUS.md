# Phase 8.1 execution status

**NUMERICAL_FAILURE.** The interrupted command survived and finished. At the start of this phase, the separate `layer0_candidate` had compiled and linked, all nine hardware fixtures had completed, and no candidate process was still running. The executable's nonzero exits came from the historical numerical gate after device execution and cleanup. The logs contain no build, link or device-launch blocker.

The candidate's nine recorded wall timings total 142 seconds, rounded. Each case took 3.190–57.700 seconds. These are whole instrumented fixture process times. The remaining operations were checkpoint replay, classification, explicit backend selection verification, and cleanup checks. Those operations completed in Phase 8.1.

The existing Phase 8 reduction run is retained in the results directory recorded in `artifacts/logs/phase8_results_path.log`. Its `candidate/device.csv` contains the previously measured 7,703 mismatch corrections. Phase 8.1 reused that evidence and did not relaunch the isolated dot-product experiment.

[Candidate hardware log](../artifacts/logs/phase8_1_hardware.log) · [Original isolated reduction log](../artifacts/logs/phase8_reduction_hardware.log) · [Source and results paths](../artifacts/logs/phase8_results_path.log)
