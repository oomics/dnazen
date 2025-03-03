default_checkpoint=/data2/peter/dnazen_pretrain_v3.1/outputs2/checkpoint-2500

if [ $1 = "--help" ]; then
    echo "Usage $0 [data_path] [output] [checkpoint] [ngram_encoder_dir]"
    exit 0
fi

data_path=$1
output=${2:-output}
checkpoint=$3
ngram_encoder_dir=$4
lr=3e-5
PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=8


# ngram_encoder_dir=/home/peter/llm_projects/ZENforDNA/resources/ngram-encoder-hg38-gue-v0.json

# epigenetic marks prediction
# for data in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
for data in H3K9ac H4 H4ac
# for data in H3K4me2 H3K14ac
do
    python run_finetune.py \
        --data_path  $data_path/GUE/EMP/$data \
        --checkpoint $checkpoint \
        --ngram_encoder_dir $ngram_encoder_dir \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
        --gradient_accumulation_steps 2 \
        --lr ${lr} \
        --num_train_epochs 4 \
        --fp16 \
        --out $output/$data 
done

# promoter detection
for data in prom_300_all prom_core_all
do
    python run_finetune.py \
        --data_path $data_path/GUE/prom/$data \
        --checkpoint $checkpoint \
        --ngram_encoder_dir $ngram_encoder_dir \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
        --gradient_accumulation_steps 2 \
        --lr $lr \
        --num_train_epochs 10 \
        --fp16 \
        --out $output/$data
done

# transcription factor prediction (human)
for data in 0 1 2 3 4
do
    python run_finetune.py \
        --data_path $data_path/GUE/tf/$data \
        --checkpoint $checkpoint \
        --ngram_encoder_dir $ngram_encoder_dir \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
        --gradient_accumulation_steps 4 \
        --lr $lr \
        --num_train_epochs 6 \
        --fp16 \
        --out $output/tf_$data
done

# transcription factor prediction (mouse)
for data in 0 1 2 3 4
do
    python run_finetune.py \
        --data_path $data_path/GUE/mouse/$data \
        --checkpoint $checkpoint \
        --ngram_encoder_dir $ngram_encoder_dir \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
        --gradient_accumulation_steps 4 \
        --lr $lr \
        --num_train_epochs 6 \
        --fp16 \
        --out $output/mouse_$data
done

# splice
for data in reconstructed
do 
    python run_finetune.py \
        --data_path $data_path/GUE/splice/$data \
        --checkpoint $checkpoint \
        --ngram_encoder_dir $ngram_encoder_dir \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
        --gradient_accumulation_steps 4 \
        --lr $lr \
        --num_train_epochs 5 \
        --fp16 \
        --out $output/splice_$data
done

# virus
for data in covid
do
    python run_finetune.py \
        --data_path $data_path/GUE/virus/$data \
        --checkpoint $checkpoint \
        --ngram_encoder_dir $ngram_encoder_dir \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
        --gradient_accumulation_steps 4 \
        --lr $lr \
        --num_train_epochs 5 \
        --fp16 \
        --out $output/virus_$data
done 