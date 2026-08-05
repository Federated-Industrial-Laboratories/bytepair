# bytepair - dependency-free byte-level BPE tokenizer library.
# Plain make, gcc or clang, Linux. `make SANITIZE=1` for ASan/UBSan.

CC      ?= gcc
CFLAGS  ?= -O3
CFLAGS  += -std=c11 -Wall -Wextra -Werror -fno-strict-aliasing -fPIC \
           -Iinclude -Isrc
LDFLAGS ?=

ifeq ($(SANITIZE),1)
CFLAGS  := $(filter-out -O3,$(CFLAGS)) -O1 -g -fsanitize=address,undefined \
           -fno-omit-frame-pointer
LDFLAGS += -fsanitize=address,undefined
endif

BUILD := build
SRC   := src/bp_util.c src/bp_vocab.c src/bp_nfc.c src/bp_scan.c \
         src/bp_scan_avx2.c src/bp_bpe.c src/bp_encode.c src/bp_census.c \
         src/tables/bp_uctables.c
OBJ   := $(SRC:%.c=$(BUILD)/%.o)

# only this translation unit is built with AVX2/BMI2; it is entered only
# after a runtime CPUID check, so the rest of the binary runs on any x86-64
$(BUILD)/src/bp_scan_avx2.o: CFLAGS += -mavx2 -mbmi2

all: $(BUILD)/libbytepair.a $(BUILD)/libbytepair.so $(BUILD)/bytepair

$(BUILD)/%.o: %.c src/bp_internal.h include/bytepair.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/libbytepair.a: $(OBJ)
	ar rcs $@ $^

$(BUILD)/libbytepair.so: $(OBJ)
	$(CC) $(LDFLAGS) -shared $^ -o $@

$(BUILD)/bytepair: cli/main.c $(BUILD)/libbytepair.a
	$(CC) $(CFLAGS) $< $(BUILD)/libbytepair.a -o $@ $(LDFLAGS) -lpthread

$(BUILD)/uctables_selftest: src/tables/bp_uctables_selftest.c \
                            $(BUILD)/src/tables/bp_uctables.o
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD)/nfc_conformance: tests/nfc_conformance.c $(BUILD)/src/bp_nfc.o \
                          $(BUILD)/src/bp_util.o \
                          $(BUILD)/src/tables/bp_uctables.o \
                          $(BUILD)/src/bp_vocab.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# fetch the pinned reference tokenizer and build the vocabulary image
vocab: $(BUILD)/qwen3.bpv
tests/data/fetched/qwen3-tokenizer.json:
	sh tests/fetch_qwen3.sh
$(BUILD)/qwen3.bpv: tests/data/fetched/qwen3-tokenizer.json tools/bpv_convert.py
	@mkdir -p $(BUILD)
	python3 tools/bpv_convert.py tests/data/fetched/qwen3-tokenizer.json $@ \
	    --source-name qwen3-tokenizer.json

check: all $(BUILD)/uctables_selftest $(BUILD)/nfc_conformance $(BUILD)/qwen3.bpv
	$(BUILD)/uctables_selftest
	sh tests/run_tests.sh

mutate: all $(BUILD)/qwen3.bpv
	sh tests/mutate.sh

clean:
	rm -rf $(BUILD)

.PHONY: all check clean vocab mutate
