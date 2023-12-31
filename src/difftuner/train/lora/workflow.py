import os
import torch 

from typing import TYPE_CHECKING, Optional, List
from pathlib import Path

from difftuner.data import get_dataset, preprocess_dataset, collate_fn
from difftuner.model import load_scheduler_and_model_and_tokenizer
from difftuner.extras.logging import get_logger
from difftuner.train.lora.trainer import CustomLoraTrainer

from diffusers.utils import is_wandb_available

from accelerate import Accelerator

from huggingface_hub import create_repo, upload_folder

if TYPE_CHECKING:
    from difftuner.hparams import ModelArguments, DataArguments, DiffusionTrainingArguemnts, FinetuningArguments


logger = get_logger(__name__)

def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
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
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def run_lora(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "DiffusionTrainingArguemnts",
    finetuning_args: "FinetuningArguments",
    accelerator: "Accelerator"
):
    dataset = get_dataset(model_args, data_args)
    noise_scheduler, unet, _, vae, text_encoder, tokenizer = load_scheduler_and_model_and_tokenizer(model_args, finetuning_args, training_args)
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, finetuning_args)

    # Dataloaders creation
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
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
    trainer = CustomLoraTrainer(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        train_dataloader=train_dataloader,
        tokenizer=tokenizer,
        accelerator=accelerator, 
        noise_scheduler=noise_scheduler, 
        unet=unet, 
        vae=vae, 
        text_encoder=text_encoder
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