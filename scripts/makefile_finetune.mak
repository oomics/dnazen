define check_defined
@ if [ -z "${$(1)}" ]; then \
	echo "Environment variable $(1) not set"; \
	exit 1; \
fi
endef

FINETUNE_LEARNING_RATE?=3e-5
PER_DEVICE_TRAIN_BATCH_SIZE?=8
PER_DEVICE_EVAL_BATCH_SIZE?=32

# build finetune extra args. make sure we have whitespace between args
FINETUNE_EXTRA_ARGS = --fp16
# check if we are running dnabert2 instead of our model
ifeq ($(DNABERT2), true)
	FINETUNE_EXTRA_ARGS+=--bert
	PRETRAIN_CHECKPOINT=zhihan1996/DNABERT-2-117M
else
	FINETUNE_EXTRA_ARGS+=--ngram_encoder_dir $(MAIN_NGRAM_ENCODER_DIR)
	PRETRAIN_CHECKPOINT=$(DNAZEN_PRETRAIN_DATA_DIR)/output/checkpoint-$(FINETUNE_CHECKPOINT_STEP)
endif

# FINETUNE_DATA_DIR = /data1/peter
check_defined_all:
	$(call check_defined,FINETUNE_DATA_DIR)
	$(call check_defined,FINETUNE_OUT_DIR)
	


$(FINETUNE_OUT_DIR)/emp/%:
	python ../src/train/run_finetune.py \
		--data_path $(FINETUNE_DATA_DIR)/GUE/EMP/$* \
		--checkpoint $(PRETRAIN_CHECKPOINT) \
		--per_device_train_batch_size $(PER_DEVICE_TRAIN_BATCH_SIZE) \
		--per_device_eval_batch_size $(PER_DEVICE_EVAL_BATCH_SIZE) \
		--gradient_accumulation_steps 1 \
		--lr $(FINETUNE_LEARNING_RATE) \
		--num_train_epochs 5 \
		$(FINETUNE_EXTRA_ARGS) \
		--out $@

$(FINETUNE_OUT_DIR)/pd/%:
		python ../src/train/run_finetune.py \
		--data_path $(FINETUNE_DATA_DIR)/GUE/prom/$* \
		--checkpoint $(PRETRAIN_CHECKPOINT) \
		--per_device_train_batch_size $(PER_DEVICE_TRAIN_BATCH_SIZE) \
		--per_device_eval_batch_size $(PER_DEVICE_EVAL_BATCH_SIZE) \
		--gradient_accumulation_steps 1 \
		--lr $(FINETUNE_LEARNING_RATE) \
		--num_train_epochs 10 \
		$(FINETUNE_EXTRA_ARGS) \
		--out $@

$(FINETUNE_OUT_DIR)/tf/%:
		python ../src/train/run_finetune.py \
		--data_path $(FINETUNE_DATA_DIR)/GUE/tf/$* \
		--checkpoint $(PRETRAIN_CHECKPOINT) \
		--per_device_train_batch_size $(PER_DEVICE_TRAIN_BATCH_SIZE) \
		--per_device_eval_batch_size $(PER_DEVICE_EVAL_BATCH_SIZE) \
		--gradient_accumulation_steps 1 \
		--lr $(FINETUNE_LEARNING_RATE) \
		--num_train_epochs 6 \
		$(FINETUNE_EXTRA_ARGS) \
		--out $@

$(FINETUNE_OUT_DIR)/mouse/%:
		python ../src/train/run_finetune.py \
		--data_path $(FINETUNE_DATA_DIR)/GUE/mouse/$* \
		--checkpoint $(PRETRAIN_CHECKPOINT) \
		--per_device_train_batch_size $(PER_DEVICE_TRAIN_BATCH_SIZE) \
		--per_device_eval_batch_size $(PER_DEVICE_EVAL_BATCH_SIZE) \
		--gradient_accumulation_steps 1 \
		--lr $(FINETUNE_LEARNING_RATE) \
		--num_train_epochs 6 \
		$(FINETUNE_EXTRA_ARGS) \
		--out $@

.PHONY: all finetune-emp finetune-pd finetune-tf check_defined_all

finetune-emp: $(FINETUNE_OUT_DIR)/emp/H3K14ac \
	$(FINETUNE_OUT_DIR)/emp/H3K4me1 \
	$(FINETUNE_OUT_DIR)/emp/H3K4me2 \
	$(FINETUNE_OUT_DIR)/emp/H3K4me3

finetune-pd: $(FINETUNE_OUT_DIR)/pd/prom_300_all \
	$(FINETUNE_OUT_DIR)/pd/prom_core_all

finetune-tf: $(FINETUNE_OUT_DIR)/tf/3 \
	$(FINETUNE_OUT_DIR)/mouse/4

all: check_defined_all finetune-pd finetune-emp finetune-tf
