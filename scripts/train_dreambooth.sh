export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="data/dog_dreambooth_test"
export CLASS_DIR="data/dog_dreambooth_test/path-to-class-images"
export OUTPUT_DIR="path-to-save-model"


CUDA_VISIBLE_DEVICES=0  accelerate launch --num_processes 1 src/train_bash.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --dreambooth_data_dir=$INSTANCE_DIR \
    --dreambooth_class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --do_train \
    --finetuning_type dreambooth \
    --dataloader_num_workers 8\
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="a photo of sks dog" \
    --class_prompt="a photo of dog" \
    --resolution=512 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=2 --gradient_checkpointing \
    --use_8bit_adam \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --warmup_steps=0 \
    --num_class_images=200 \
    --max_steps=800 \
    --report_to wandb \
    --mixed_precision no\
    --validation_prompt "a photo of sks dog in a bucket"