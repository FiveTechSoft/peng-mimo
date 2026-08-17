UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
# --- macOS / Apple Silicon ---
# Apple clang non include il runtime OpenMP: se c'e' libomp di Homebrew lo usa
# (brew install libomp), altrimenti compila single-thread (i pragma omp sono ignorati).
# Niente -march: su arm64 NEON e' baseline (i kernel __ARM_NEON si attivano da soli).
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(OMPDIR),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp non trovato: build SINGLE-THREAD. Per il multithread: brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(OMPC) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm $(OMPL) -pthread -lzstd
else
# --- Linux x86-64 (percorso originale, invariato) ---
CC      = gcc
# ARCH=native -> ottimizzato per QUESTA macchina (default, piu' veloce).
# ARCH=x86-64-v3 -> binario PORTABILE su qualsiasi x86-64 moderno con AVX2 (per distribuire).
# ARCH=x86-64 -> massima compatibilita' (niente AVX2: usa il path scalare di fallback).
ARCH   ?= native
CFLAGS  = -O3 -g -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -pthread -rdynamic -lzstd
endif

# CUDA=1 adds an opt-in backend for resident tensors. The default build remains
# pure C and keeps the original zero-dependency runtime.
CUDA      ?= 0
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
# -U_GNU_SOURCE: glibc >= 2.41 declara las funciones math C23 (cospi/rsqrt/...)
# con noexcept(true) cuando _GNU_SOURCE; crt/math_functions.h de CUDA 12.8 las
# declara sin noexcept -> conflicto duro. Sin _GNU_SOURCE glibc no las declara.
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra -U_GNU_SOURCE
PYTHON    ?= python3
CUDA_OBJ  =
TEST_BINS = tests/test_json tests/test_st tests/test_tier tests/test_gateup tests/test_topp_heap
ifeq ($(CUDA),1)
ifeq ($(UNAME_S),Darwin)
$(error CUDA=1 is supported only on Linux)
endif
CFLAGS   += -DCOLI_CUDA
LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lstdc++
CUDA_OBJ  = backend_cuda.o
endif

all: mimo

glm: glm.c st.h json.h tok.h tok_unicode.h compat.h $(CUDA_OBJ)
	$(CC) $(CFLAGS) glm.c $(CUDA_OBJ) -o glm $(LDFLAGS)

mimo: mimo.c st.h json.h tok.h tok_unicode.h compat.h $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) mimo.c $(CUDA_OBJ) -o mimo $(LDFLAGS)

backend_cuda.o: backend_cuda.cu backend_cuda.h
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) -c backend_cuda.cu -o $@

cuda-test: backend_cuda.cu backend_cuda.h tests/test_backend_cuda.cu
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) backend_cuda.cu tests/test_backend_cuda.cu -o backend_cuda_test
	./backend_cuda_test

# kernel-level A/B del path MoE decode (sin modelo): fused_down_acc, pinned staging
fused-bench: backend_cuda.cu backend_cuda.h tests/fused_down_bench.cu
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) backend_cuda.cu tests/fused_down_bench.cu -o fused_down_bench

olmoe: olmoe.c st.h json.h
	$(CC) $(CFLAGS) olmoe.c -o olmoe $(LDFLAGS)

# binario portabile da distribuire su altre macchine x86-64 (peng: mimo)
portable:
	$(MAKE) mimo ARCH=x86-64-v3

tests/test_json: tests/test_json.c json.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st: tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tier: tests/test_tier.c tier.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

tests/test_gateup: tests/test_gateup.c mimo.c st.h tok.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

tests/test_topp_heap: tests/test_topp_heap.c mimo.c st.h tok.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	@for test in $(TEST_BINS); do ./$$test || exit 1; done

test-python:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

# Dependency-light: build mimo + C tests + API template tests (no torch).
check-core: mimo test-c
	$(PYTHON) -m unittest tests.test_openai_server tests.test_template_overflow -v

test: test-c test-python

# Local validation: portable mimo + full test suite (needs Python deps for some tests).
check:
	$(MAKE) clean
	$(MAKE) portable
	$(MAKE) test

clean:
	rm -f olmoe glm mimo backend_cuda.o backend_cuda_test $(TEST_BINS)
	rm -rf tests/__pycache__

.PHONY: all cuda-test portable test-c test-python test check check-core clean
