CUDA_VISIBLE_DEVICES=0  accelerate launch --num_processes 1 src/train_bash.py \
    --pretrained_model_name_or_path /mnt/ssd-array/xx-volume/develop/MLLM/stable-diffusion-webui/models/Stable-diffusion/stable-diffusion-xl-base-1.0/ \
    --train_data_dir /mnt/ssd-array/xx-volume/develop/MLLM/Diffusion-Factory/data/dld_dataset\
    --do_train \
    --finetuning_type sdxl-lora \
    --dataloader_num_workers 8\
    --output_dir sdxl-dld-model \
    --resolution 512  --center_crop --random_flip \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --max_steps 2000 \
    --report_to wandb \
    --validation_prompt "donglongdong style, a beautiful girl"\

    #--train_data_dir /mnt/ssd-array/xx-volume/develop/MLLM/Diffusion-Factory/data/anime_dataset \
