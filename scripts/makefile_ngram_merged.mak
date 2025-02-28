
define check_defined
@ if [ -z "${$(1)}" ]; then \
	echo "Environment variable $(1) not set"; \
	exit 1; \
fi
endef

ifneq ($(filter $(SIDE_NGRAM_ENCODER_DIR), $(wildcard $(NGRAM_ENCODER_DIR)/ngram-encoder-%.json)),)
	$(error "side ngram should be of format NGRAM_ENCODER_DIR/ngram-encoder%.json" )
endif

DNAZEN_PRETRAIN_PER_DEVICE_BATCH_SIZE = 128
DNAZEN_PRETRAIN_GRAD_ACCU_STEPS = 4

NGRAM_MAX_LEN ?= 5

.PHONY: pretrain_data pretrain check_defined_all

check_defined_all:
# all ngrams
	$(call check_defined,ALL_NGRAM_ENCODERS)
# the NGRAM training method (pmi for freq)
	$(call check_defined,NGRAM_TRAIN_METHOD)
# path to the ngram encoder used in training
	$(call check_defined,MAIN_NGRAM_ENCODER_DIR)
# path to the ngram encoder used for making core ngrams
	$(call check_defined,SIDE_NGRAM_ENCODER_DIR)
# path to the core ngram directory
	$(call check_defined,CORE_NGRAM_DIR)

# DATA SOURCE TO TRAIN NGRAM ENCODER
	$(call check_defined,MAIN_NGRAM_ENCODER_DATA_SOURCES)

# --- hyper params ---
	$(call check_defined,NGRAM_MIN_NGRAM_FREQ)
	$(call check_defined,NGRAM_MIN_TOKEN_FREQ)
	$(call check_defined,NGRAM_MIN_PMI)
	$(call check_defined,DNAZEN_PRETRAIN_LR)


$(NGRAM_ENCODER_DIR)/ngram-encoder-%.json:
	python train_ngram_encoder.py \
		-d $* \
		-o $@ \
		--min-ngram $(NGRAM_MIN_NGRAM_FREQ) \
		--min-tok $(NGRAM_MIN_TOKEN_FREQ) \
		--min-pmi $(NGRAM_MIN_PMI) \
		--max-ngram-len $(NGRAM_MAX_LEN) \
		--train-method $(NGRAM_TRAIN_METHOD)

$(CORE_NGRAM_DIR): $(SIDE_NGRAM_ENCODER_DIR) # side ngram should be one of NGRAM_ENCODER_DIR/ngram-encoder%.json
	python make_core_ngrams.py \
		--ngram_file1 $(SIDE_NGRAM_ENCODER_DIR) \
		--out $@

_MERGED_NGRAM_ARGS=$(foreach target,$(ALL_NGRAM_ENCODERS),--ngram-files $(target))
$(MAIN_NGRAM_ENCODER_DIR): $(ALL_NGRAM_ENCODERS) # should be NGRAM_ENCODER_DIR/ngram-encoder%.json separated by space
	python make_merged_ngram_encoder.py \
		$(_MERGED_NGRAM_ARGS) \
		--out $@

all: check_defined_all $(CORE_NGRAM_DIR) $(MAIN_NGRAM_ENCODER_DIR)
