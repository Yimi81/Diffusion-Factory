CUDA_VISIBLE_DEVICES=0  accelerate launch --num_processes 1 src/train_bash.py \
    --model_name_or_path runwayml/stable-diffusion-v1-5 \
    --train_data_dir /mnt/ssd-array/xx-volume/develop/MLLM/Diffusion-Factory/data/hyhh_dataset \
    --do_train \
    --finetuning_type lora \
    --dataloader_num_workers 8\
    --output_dir hyhh-model \
    --resolution 512  --center_crop --random_flip \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --max_steps 2000 \
    --report_to wandb \
    --validation_prompt "hyhh style, tree on the mountain."\

    #--train_data_dir /mnt/ssd-array/xx-volume/develop/MLLM/Diffusion-Factory/data/anime_dataset \
