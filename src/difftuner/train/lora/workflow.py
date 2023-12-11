import os
import math
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path
import numpy as np

import torch 
import torch.nn.functional as F
import shutil

from packaging import version
from tqdm.auto import tqdm

from diffusers.optimization import get_scheduler

from difftuner.data import get_dataset, preprocess_dataset, collate_fn
from difftuner.model import load_scheduler_and_model_and_tokenizer
from difftuner.extras.logging import get_logger
from difftuner.train.lora.trainer import CustomLoraTrainer

from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from accelerate import Accelerator

from huggingface_hub import create_repo, upload_folder

if TYPE_CHECKING:
    from difftuner.hparams import ModelArguments, DataArguments, DiffusionTrainingArguemnts, FinetuningArguments

if is_wandb_available():
    import wandb


logger = get_logger(__name__)

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
        trainer.train()

    # Predict
    if training_args.do_predict:
        trainer.predict()

