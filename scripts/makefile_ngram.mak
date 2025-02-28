define check_defined
@ if [ -z "${$(1)}" ]; then \
	echo "Environment variable $(1) not set"; \
	exit 1; \
fi
endef

check_defined_all:
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

$(MAIN_NGRAM_ENCODER_DIR):
	python train_ngram_encoder.py \
		-d $(MAIN_NGRAM_ENCODER_DATA_SOURCES) \
		-o $(MAIN_NGRAM_ENCODER_DIR) \
		--min-ngram $(NGRAM_MIN_NGRAM_FREQ) \
		--min-tok $(NGRAM_MIN_TOKEN_FREQ) \
		--min-pmi $(NGRAM_MIN_PMI) \
		--train-method $(NGRAM_TRAIN_METHOD)

$(SIDE_NGRAM_ENCODER_DIR):
	python train_ngram_encoder.py \
		-d gue_all  \
		-o $(SIDE_NGRAM_ENCODER_DIR) \
		--min-ngram $(NGRAM_MIN_NGRAM_FREQ) \
		--min-tok $(NGRAM_MIN_TOKEN_FREQ) \
		--min-pmi $(NGRAM_MIN_PMI) \
		--train-method $(NGRAM_TRAIN_METHOD)

$(CORE_NGRAM_DIR): $(MAIN_NGRAM_ENCODER_DIR) $(SIDE_NGRAM_ENCODER_DIR)
	python make_core_ngrams.py \
		--ngram_file1 $(MAIN_NGRAM_ENCODER_DIR) \
		--ngram_file2 $(SIDE_NGRAM_ENCODER_DIR) \
		--out $(CORE_NGRAM_DIR)

.PHONY check_defined_all core_ngram main_ngram

core_ngram: check_defined_all $(CORE_NGRAM_DIR)

main_ngram: check_defined_all $(MAIN_NGRAM_ENCODER_DIR)