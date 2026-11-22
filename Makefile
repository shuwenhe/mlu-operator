SHELL := /usr/bin/env bash

MODEL_DIR ?= $(abspath ../model/DeepSeek-V2-Lite)
TP ?= 8
MAX_TOKENS ?= 32
DTYPE ?= bfloat16
FROZEN_IMAGE ?= sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756
NEUWARE_ROOT ?= /usr/local/neuware
CARD ?= 0
WARM_REQUESTS ?= 2
MAX_MODEL_LEN ?= 128
TEMPERATURE ?= 0.7

DEEPSEEK_DEFAULT_VARIANT ?= stage_c3
DEEPSEEK_DEFAULT_EXPERIMENT ?= fused,batched
DEEPSEEK_TP8_VARIANT ?= stage_c1
DEEPSEEK_TP8_EXPERIMENT ?= baseline
DEEPSEEK_C2_VARIANT ?= stage_c2
DEEPSEEK_C2_EXPERIMENT ?= output-axis
DEEPSEEK_C3_VARIANT ?= stage_c3
DEEPSEEK_C3_EXPERIMENT ?= fused,batched

BUILD_CPU_DIR := build/cpu
BUILD_MLU_DIR := build/mlu
BUILD_NATIVE_DIR := build/native

.PHONY: help check configure build cpu mlu native test clean \
	deepseek deepseek-check deepseek-verify deepseek-c2 deepseek-c3 deepseek-tp8 \
	benchmark deepseek-benchmark moe-audit stage-c1 stage-c2 stage-c3 deepseek-chat

help:
	@echo "Available targets:"
	@echo "  make check                  - Environment diagnostics (host + SDK + devices)"
	@echo "  make configure              - CMake configure for BUILD_DIR (default build/cpu)"
	@echo "  make build                  - Build configured BUILD_DIR"
	@echo "  make cpu                    - Build/test CPU reference path"
	@echo "  make mlu                    - Build MLU objects (no SDK link)"
	@echo "  make native                 - Build SDK-linked native binaries"
	@echo "  make test                   - Run CTest in build/cpu"
	@echo "  make clean                  - Remove build directories"
	@echo ""
	@echo "  make deepseek-verify        - Verify frozen C.1–C.3 source and evidence"
	@echo "  make deepseek-check         - Validate image/model/MLU devices before inference"
	@echo "  make deepseek               - Run real TP8 inference (default Stage C.3 fused,batched)"
	@echo "  make deepseek-c2            - Run real TP8 inference with Stage C.2 model"
	@echo "  make deepseek-c3            - Run real TP8 inference with Stage C.3 model"
	@echo "  make deepseek-tp8           - Run real TP8 inference with Stage C.1 TP8 model"
	@echo "  make deepseek-benchmark     - Run explicit Stage C.3 benchmark (fused,batched)"
	@echo "  make deepseek-chat          - Run interactive Stage C.3 TP8 chat (single engine)"
	@echo ""
	@echo "  make benchmark BENCH_AUTHORIZED=1 CARD=<0-7>  - Run native operator benchmark"
	@echo "  make moe-audit              - Run existing Stage C.3 eight-rank probe"
	@echo "  make stage-c1               - Run the Stage C.1 model through the product launcher"
	@echo "  make stage-c2               - Run existing Stage C.2 candidate benchmark"
	@echo "  make stage-c3               - Run existing Stage C.3 candidate benchmark"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL_DIR=$(MODEL_DIR)"
	@echo "  TP=$(TP) MAX_TOKENS=$(MAX_TOKENS) MAX_MODEL_LEN=$(MAX_MODEL_LEN) DTYPE=$(DTYPE)"
	@echo "  TEMPERATURE=$(TEMPERATURE)"
	@echo "  FROZEN_IMAGE=$(FROZEN_IMAGE)"
	@echo "  WARM_REQUESTS=$(WARM_REQUESTS)"

check:
	@scripts/environment_check.sh
	@test -d "$(MODEL_DIR)" || (echo "Model directory missing: $(MODEL_DIR)" >&2; exit 2)
	@docker image inspect "$(FROZEN_IMAGE)" >/dev/null 2>&1 || (echo "Frozen image missing: $(FROZEN_IMAGE)" >&2; exit 2)
	@echo "check: PASS"

configure:
	@BUILD_DIR="$${BUILD_DIR:-$(BUILD_CPU_DIR)}"; \
	cmake -S . -B "$$BUILD_DIR"

build:
	@BUILD_DIR="$${BUILD_DIR:-$(BUILD_CPU_DIR)}"; \
	cmake --build "$$BUILD_DIR" --parallel

cpu:
	@cmake -S . -B "$(BUILD_CPU_DIR)"
	@cmake --build "$(BUILD_CPU_DIR)" --parallel
	@cd "$(BUILD_CPU_DIR)" && ctest --output-on-failure

mlu:
	@cmake -S . -B "$(BUILD_MLU_DIR)" -DMLU_NATIVE=ON -DMLU_LINK=OFF -DNEUWARE_ROOT="$(NEUWARE_ROOT)"
	@cmake --build "$(BUILD_MLU_DIR)" --parallel

native:
	@cmake -S . -B "$(BUILD_NATIVE_DIR)" -DMLU_NATIVE=ON -DMLU_LINK=ON -DNEUWARE_ROOT="$(NEUWARE_ROOT)"
	@cmake --build "$(BUILD_NATIVE_DIR)" --parallel

test:
	@cd "$(BUILD_CPU_DIR)" && ctest --output-on-failure
	@scripts/test_deepseek_routing.sh
	@scripts/test_deepseek_chat_mock.sh

clean:
	@rm -rf "$(BUILD_CPU_DIR)" "$(BUILD_MLU_DIR)" "$(BUILD_NATIVE_DIR)"

deepseek-verify:
	@sha256sum -c --quiet artifacts/logs/phase15/stage_c1/source_hashes.txt
	@sha256sum -c --quiet artifacts/logs/phase15/stage_c2/{source_hashes,stage_c1_evidence_hashes}.txt
	@sha256sum -c --quiet artifacts/logs/phase15/stage_c3/{sha256_manifest,stage_c2_evidence_hashes}.txt
	@echo "Stage C.1–C.3 source and evidence checksums verified."

deepseek-check:
	@MODEL_DIR="$(MODEL_DIR)" TP="$(TP)" DTYPE="$(DTYPE)" MAX_TOKENS="$(MAX_TOKENS)" \
	VARIANT="$(DEEPSEEK_DEFAULT_VARIANT)" DEEPSEEK_EXPERIMENT="$(DEEPSEEK_DEFAULT_EXPERIMENT)" \
	FROZEN_IMAGE="$(FROZEN_IMAGE)" WARM_REQUESTS="$(WARM_REQUESTS)" \
	scripts/deepseek_infer.sh "$@"

deepseek:
	@MODEL_DIR="$(MODEL_DIR)" TP="$(TP)" DTYPE="$(DTYPE)" MAX_TOKENS="$(MAX_TOKENS)" \
	VARIANT="$(DEEPSEEK_DEFAULT_VARIANT)" DEEPSEEK_EXPERIMENT="$(DEEPSEEK_DEFAULT_EXPERIMENT)" \
	FROZEN_IMAGE="$(FROZEN_IMAGE)" WARM_REQUESTS="$(WARM_REQUESTS)" \
	scripts/deepseek_infer.sh "$@"

deepseek-tp8:
	@MODEL_DIR="$(MODEL_DIR)" TP="$(TP)" DTYPE="$(DTYPE)" MAX_TOKENS="$(MAX_TOKENS)" \
	VARIANT="$(DEEPSEEK_TP8_VARIANT)" DEEPSEEK_EXPERIMENT="$(DEEPSEEK_TP8_EXPERIMENT)" \
	FROZEN_IMAGE="$(FROZEN_IMAGE)" WARM_REQUESTS="$(WARM_REQUESTS)" \
	scripts/deepseek_infer.sh "$@"

deepseek-c2:
	@MODEL_DIR="$(MODEL_DIR)" TP="$(TP)" DTYPE="$(DTYPE)" MAX_TOKENS="$(MAX_TOKENS)" \
	VARIANT="$(DEEPSEEK_C2_VARIANT)" DEEPSEEK_EXPERIMENT="$(DEEPSEEK_C2_EXPERIMENT)" \
	FROZEN_IMAGE="$(FROZEN_IMAGE)" WARM_REQUESTS="$(WARM_REQUESTS)" \
	scripts/deepseek_infer.sh "$@"

deepseek-c3:
	@MODEL_DIR="$(MODEL_DIR)" TP="$(TP)" DTYPE="$(DTYPE)" MAX_TOKENS="$(MAX_TOKENS)" \
	VARIANT="$(DEEPSEEK_C3_VARIANT)" DEEPSEEK_EXPERIMENT="$(DEEPSEEK_C3_EXPERIMENT)" \
	FROZEN_IMAGE="$(FROZEN_IMAGE)" WARM_REQUESTS="$(WARM_REQUESTS)" \
	scripts/deepseek_infer.sh "$@"

deepseek-benchmark:
	@MODEL_DIR="$(MODEL_DIR)" TP="$(TP)" DTYPE="$(DTYPE)" MAX_TOKENS="$(MAX_TOKENS)" \
	VARIANT="$(DEEPSEEK_C3_VARIANT)" DEEPSEEK_EXPERIMENT="$(DEEPSEEK_C3_EXPERIMENT)" \
	FROZEN_IMAGE="$(FROZEN_IMAGE)" WARM_REQUESTS="$(WARM_REQUESTS)" \
	scripts/deepseek_infer.sh "$@"

moe-audit:
	@MODEL_DIR="$(MODEL_DIR)" TP="$(TP)" DTYPE="$(DTYPE)" MAX_TOKENS="$(MAX_TOKENS)" \
	VARIANT="$(DEEPSEEK_C3_VARIANT)" DEEPSEEK_EXPERIMENT="$(DEEPSEEK_C3_EXPERIMENT)" FROZEN_IMAGE="$(FROZEN_IMAGE)" \
	WARM_REQUESTS="$(WARM_REQUESTS)" scripts/deepseek_infer.sh "$@"

stage-c2:
	@MODEL_DIR="$(MODEL_DIR)" TP="$(TP)" DTYPE="$(DTYPE)" MAX_TOKENS="$(MAX_TOKENS)" \
	VARIANT="$(DEEPSEEK_C2_VARIANT)" DEEPSEEK_EXPERIMENT="$(DEEPSEEK_C2_EXPERIMENT)" \
	FROZEN_IMAGE="$(FROZEN_IMAGE)" WARM_REQUESTS="$(WARM_REQUESTS)" scripts/deepseek_infer.sh "$@"

stage-c3:
	@MODEL_DIR="$(MODEL_DIR)" TP="$(TP)" DTYPE="$(DTYPE)" MAX_TOKENS="$(MAX_TOKENS)" \
	VARIANT="$(DEEPSEEK_C3_VARIANT)" DEEPSEEK_EXPERIMENT="$(DEEPSEEK_C3_EXPERIMENT)" \
	FROZEN_IMAGE="$(FROZEN_IMAGE)" WARM_REQUESTS="$(WARM_REQUESTS)" scripts/deepseek_infer.sh "$@"

stage-c1: deepseek-tp8

deepseek-chat:
	@MODEL_DIR="$(MODEL_DIR)" TP="$(TP)" DTYPE="$(DTYPE)" MAX_TOKENS="$(MAX_TOKENS)" \
	MAX_MODEL_LEN="$(MAX_MODEL_LEN)" TEMPERATURE="$(TEMPERATURE)" HISTORY_POLICY="$(HISTORY_POLICY)" \
	VARIANT="$(DEEPSEEK_C3_VARIANT)" DEEPSEEK_EXPERIMENT="$(DEEPSEEK_C3_EXPERIMENT)" \
	FROZEN_IMAGE="$(FROZEN_IMAGE)" scripts/deepseek_chat.sh

benchmark:
	@if [[ "$${BENCH_AUTHORIZED:-0}" != "1" ]]; then \
		echo "Refusing benchmark: set BENCH_AUTHORIZED=1 and CARD=<0-7> explicitly." >&2; \
		exit 2; \
	fi
	@NEUWARE_ROOT="$(NEUWARE_ROOT)" scripts/benchmark.sh --authorized "$(CARD)"

