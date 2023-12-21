import os
import math
from typing import TYPE_CHECKING
from pathlib import Path

import torch 

from diffusers.optimization import get_scheduler

from difftuner.data import get_dataset, sdxl_preprocess_dataset, sdxl_collate_fn
from difftuner.model import sdxl_load_scheduler_and_model_and_tokenizer
from difftuner.extras.logging import get_logger
from difftuner.train.lora.sdxl.trainer import SDXLCustomLoraTrainer

from accelerate import Accelerator

from huggingface_hub import create_repo, upload_folder

if TYPE_CHECKING:
    from difftuner.hparams import ModelArguments, DataArguments, DiffusionTrainingArguemnts, FinetuningArguments

logger = get_logger(__name__)

def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    dataset_name=str,
    train_text_encoder=False,
    repo_folder=None,
    vae_path=None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
dataset: {dataset_name}
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def run_sdxl_lora(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "DiffusionTrainingArguemnts",
    finetuning_args: "FinetuningArguments",
    accelerator: "Accelerator"
):
    dataset = get_dataset(model_args, data_args)
    noise_scheduler, unet, _, vae, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two = sdxl_load_scheduler_and_model_and_tokenizer(model_args, finetuning_args, training_args)
    dataset = sdxl_preprocess_dataset(dataset, tokenizer_one, tokenizer_two, data_args, training_args, finetuning_args)

    # Dataloaders creation
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=sdxl_collate_fn,
        batch_size=training_args.per_device_train_batch_size ,
        num_workers=training_args.dataloader_num_workers
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)

        if training_args.push_to_hub:
            repo_id = create_repo(
                repo_id=training_args.hub_model_id or Path(training_args.output_dir).name, exist_ok=True, token=training_args.hub_token
            ).repo_id

    # Initialize our Trainer
    trainer = SDXLCustomLoraTrainer(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        train_dataloader=train_dataloader,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_one,
        accelerator=accelerator, 
        noise_scheduler=noise_scheduler, 
        unet=unet, 
        vae=vae, 
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two
    )

    # Training and Evaluation
    if training_args.do_train:
        images = trainer.train()

    # Predict
    if training_args.do_predict:
        images = trainer.predict()

    # Create model card
    if training_args.push_to_hub:
        save_model_card(
            repo_id,
            images=images,
            base_model=model_args.pretrained_model_name_or_path,
            dataset_name=data_args.dataset_name,
            repo_folder=training_args.output_dir,
        )
        upload_folder(
            repo_id=repo_id,
            folder_path=training_args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )