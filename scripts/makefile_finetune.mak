define check_defined
@ if [ -z "${$(1)}" ]; then \
	echo "Environment variable $(1) not set"; \
	exit 1; \
fi
endef

FINETUNE_LEARNING_RATE?=3e-5
PER_DEVICE_TRAIN_BATCH_SIZE?=8
PER_DEVICE_EVAL_BATCH_SIZE?=32
FINETUNE_OUT_DIR?=../data/finetune-results

ALL_EMP_TASK_NAMES?=H3K14ac H3K4me1 H3K4me2 H3K4me3
ALL_PROM_DETECTION_TASK_NAMES?=prom_300_all prom_core_all
ALL_TRANS_FACTORS_TASK_NAMES?=0 1 2 3 4
ALL_TRANS_FACTORS_MOUSE_TASK_NAMES?=3 4

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
# $(call check_defined,FINETUNE_OUT_DIR)

$(FINETUNE_OUT_DIR)/EMP/%:
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

$(FINETUNE_OUT_DIR)/prom/%:
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

_EMP_TASKS=$(foreach target,$(ALL_EMP_TASK_NAMES),$(FINETUNE_OUT_DIR)/EMP/$(target))
finetune-emp: $(_EMP_TASKS)
# finetune-emp: $(FINETUNE_OUT_DIR)/EMP/H3K14ac \
# 	$(FINETUNE_OUT_DIR)/EMP/H3K4me1 \
# 	$(FINETUNE_OUT_DIR)/EMP/H3K4me2 \
# 	$(FINETUNE_OUT_DIR)/EMP/H3K4me3

_PROM_DETECTION_TASKS=$(foreach target,$(ALL_PROM_DETECTION_TASK_NAMES),$(FINETUNE_OUT_DIR)/prom/$(target))
finetune-pd: $(_PROM_DETECTION_TASKS)
# finetune-pd: $(FINETUNE_OUT_DIR)/prom/prom_300_all \
# 	$(FINETUNE_OUT_DIR)/prom/prom_core_all

_TRANS_FACTORS_TASKS=$(foreach target,$(ALL_TRANS_FACTORS_TASK_NAMES),$(FINETUNE_OUT_DIR)/tf/$(target))
finetune-tf: $(_TRANS_FACTORS_TASKS)

_TRANS_FACTORS_MOUSE_TASKS=$(foreach target,$(ALL_TRANS_FACTORS_MOUSE_TASK_NAMES),$(FINETUNE_OUT_DIR)/mouse/$(target))
finetune-mouse: $(_TRANS_FACTORS_MOUSE_TASKS)
# finetune-mouse: $(FINETUNE_OUT_DIR)/mouse/3 \
# 	$(FINETUNE_OUT_DIR)/mouse/4

all: check_defined_all finetune-pd finetune-emp finetune-tf finetune-mouse
