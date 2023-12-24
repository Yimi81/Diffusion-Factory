export MODEL_NAME="/mnt/ssd-array/xx-volume/develop/MLLM/stable-diffusion-webui/models/Stable-diffusion/stable-diffusion-xl-base-1.0/"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

CUDA_VISIBLE_DEVICES=0  accelerate launch --num_processes 1  src/train_bash.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --do_train \
  --finetuning_type sdxl-lora \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --per_device_train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="a cute Sundar Pichai creature" \
  --validation_epochs 5 \
  --save_steps=5000 \
  --output_dir="sdxl-pokemon-model" \
  --enable_xformers_memory_efficient_attention=False\