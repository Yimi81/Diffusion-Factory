import os
from typing import TYPE_CHECKING
from pathlib import Path

import torch 

from difftuner.data import get_dataset, preprocess_dataset, dreambooth_collate_fn
from difftuner.model import load_scheduler_and_model_and_tokenizer
from difftuner.train.dreambooth.trainer import CustomDreamBoothTrainer

from accelerate import Accelerator

from difftuner.extras.logging import get_logger

from huggingface_hub import create_repo, upload_folder

if TYPE_CHECKING:
    from difftuner.hparams import ModelArguments, DataArguments, DiffusionTrainingArguemnts, FinetuningArguments


logger = get_logger(__name__)


def run_dreambooth(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "DiffusionTrainingArguemnts",
    finetuning_args: "FinetuningArguments",
    accelerator: "Accelerator"
):
    dataset = get_dataset(model_args, data_args)
    noise_scheduler, unet, _, vae, text_encoder, tokenizer = load_scheduler_and_model_and_tokenizer(model_args, finetuning_args, training_args)
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, finetuning_args)

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (sayakpaul): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if training_args.train_text_encoder and training_args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )
    
    # Dataloaders creation
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda examples: dreambooth_collate_fn(examples, training_args.with_prior_preservation),
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
    trainer = CustomDreamBoothTrainer(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        train_dataloader=train_dataloader,
        tokenizer=tokenizer,
        accelerator=accelerator, 
        noise_scheduler=noise_scheduler, 
        unet=unet, 
        controlnet=None, 
        vae=vae, 
        text_encoder=text_encoder
    )

    # Training and Evaluation
    if training_args.do_train:
        trainer.train()

    # Predict
    if training_args.do_predict:
        trainer.predict()