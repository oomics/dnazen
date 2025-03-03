define check_defined
@ if [ -z "${$(1)}" ]; then \
	echo "Environment variable $(1) not set"; \
	exit 1; \
fi
endef

DNAZEN_PRETRAIN_PER_DEVICE_BATCH_SIZE = 128
DNAZEN_PRETRAIN_GRAD_ACCU_STEPS = 4

.PHONY: pretrain_data pretrain check_defined_all

check_defined_all:
# path to raw pre-train data directory
	$(call check_defined,DNAZEN_PRETRAIN_RAW_DATA_DIR)
# path to pre-train data directory
	$(call check_defined,DNAZEN_PRETRAIN_DATA_DIR)
# the NGRAM training method (pmi for freq)
	$(call check_defined,NGRAM_TRAIN_METHOD)
# path to the ngram encoder used in training
	$(call check_defined,MAIN_NGRAM_ENCODER_DIR)
# path to the ngram encoder used for making core ngrams
	$(call check_defined,SIDE_NGRAM_ENCODER_DIR)
# path to the core ngram directory
	$(call check_defined,CORE_NGRAM_DIR)
# path to finetune results and checkpoints
	$(call check_defined,FINETUNE_OUT_DIR)
# path to finetune data directory
	$(call check_defined,FINETUNE_DATA_DIR)
# the stepsize of pretrained model. Used as base model for further pretraining.
	$(call check_defined,FINETUNE_CHECKPOINT_STEP)

# DATA SOURCE TO TRAIN NGRAM ENCODER
	$(call check_defined,MAIN_NGRAM_ENCODER_DATA_SOURCES)

# --- hyper params ---
	$(call check_defined,NGRAM_MIN_NGRAM_FREQ)
	$(call check_defined,NGRAM_MIN_TOKEN_FREQ)
	$(call check_defined,NGRAM_MIN_PMI)
	$(call check_defined,DNAZEN_PRETRAIN_LR)

DNAZEN_RAW_TRAIN_DATASET=$(DNAZEN_PRETRAIN_RAW_DATA_DIR)/train.txt
DNAZEN_RAW_VALID_DATASET=$(DNAZEN_PRETRAIN_RAW_DATA_DIR)/dev.txt
DNAZEN_TOKENIZED_TRAIN_DATASET_DIR=$(DNAZEN_PRETRAIN_RAW_DATA_DIR)/train.pt
DNAZEN_TOKENIZED_VALID_DATASET_DIR=$(DNAZEN_PRETRAIN_RAW_DATA_DIR)/dev.pt
$(DNAZEN_TOKENIZED_TRAIN_DATASET_DIR):
	python make_tokenized_dataset.py \
		--data $(DNAZEN_RAW_TRAIN_DATASET) \
		-o $@

$(DNAZEN_TOKENIZED_VALID_DATASET_DIR):
	python make_tokenized_dataset.py \
		--data $(DNAZEN_RAW_VALID_DATASET) \
		-o $@

# feat: use merged ngrams instead
$(DNAZEN_PRETRAIN_DATA_DIR): \
	$(CORE_NGRAM_DIR) \
	$(MAIN_NGRAM_ENCODER_DIR) \
	$(DNAZEN_TOKENIZED_VALID_DATASET_DIR) \
	$(DNAZEN_TOKENIZED_TRAIN_DATASET_DIR)
	python make_pretrain_dataset.py \
		--data-source tokenized \
		-d $(DNAZEN_PRETRAIN_RAW_DATA_DIR) \
		--ngram $(MAIN_NGRAM_ENCODER_DIR) \
		--core-ngram $(CORE_NGRAM_DIR) \
		--max-ngrams 30 \
		--out $@

PRETRAIN_TRAIN_DATA_DIR=$(DNAZEN_PRETRAIN_DATA_DIR)/train
PRETRAIN_VALID_DATA_DIR=$(DNAZEN_PRETRAIN_DATA_DIR)/dev
PRETRAIN_OUT_DIR=$(DNAZEN_PRETRAIN_DATA_DIR)/output
$(PRETRAIN_OUT_DIR): $(DNAZEN_PRETRAIN_DATA_DIR)
	python run_pretrain.py \
		--train $(PRETRAIN_TRAIN_DATA_DIR) \
		--dev $(PRETRAIN_VALID_DATA_DIR) \
		--out $@ \
		--lr $(DNAZEN_PRETRAIN_LR) \
		--per-device-train-batch-size $(DNAZEN_PRETRAIN_PER_DEVICE_BATCH_SIZE) \
		--grad-accumulation-steps $(DNAZEN_PRETRAIN_GRAD_ACCU_STEPS)

main_ngram:
# if you would like to run older experiments, switch to `makefile_ngram.mak`.
# In the older experiments the main ngram encoder was made in another method.
	make -f makefile_ngram_merged.mak all 
	@echo "Ngram ready at $(MAIN_NGRAM_ENCODER_DIR)"

pretrain_data: $(DNAZEN_PRETRAIN_DATA_DIR)
	@echo "Pretrain data is ready at $(DNAZEN_PRETRAIN_DATA_DIR)"

pretrain: check_defined_all main_ngram $(PRETRAIN_OUT_DIR)
	@echo "Pretraining done. Output is at $(PRETRAIN_OUT_DIR)"

finetune:
	make -f makefile_finetune.mak all
	@echo "Finetuning done. Output is at $(FINETUNE_OUT_DIR)"
