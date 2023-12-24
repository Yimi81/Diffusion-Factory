CUDA_VISIBLE_DEVICES=0  accelerate launch --num_processes 1 src/train_bash.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --train_data_dir /mnt/ssd-array/xx-volume/develop/MLLM/Diffusion-Factory/data/dld_dataset\
    --do_train \
    --finetuning_type lora \
    --dataloader_num_workers 8\
    --output_dir dld-model \
    --resolution 512  --center_crop --random_flip \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --max_steps 2000 \
    --report_to wandb \
    --validation_prompt "donglongdong style, outdoors, smile, brown hair, brown eyes, looking at viewer, short hair, mountain, coat, holding, solo, 1girl."\

    #--train_data_dir /mnt/ssd-array/xx-volume/develop/MLLM/Diffusion-Factory/data/anime_dataset \
