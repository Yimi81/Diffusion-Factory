import os
import math
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path
import numpy as np

import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import shutil

from packaging import version
from tqdm.auto import tqdm

from transformers import PreTrainedTokenizer, CLIPTextModel, CLIPTokenizer, AutoTokenizer

from difftuner.data import get_dataset, preprocess_dataset, collate_fn
from difftuner.model import load_scheduler_and_model_and_tokenizer
from difftuner.extras.logging import get_logger

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, ControlNetModel, DiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.models.lora import LoRALinearLayer
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import accelerate
from accelerate import Accelerator

from huggingface_hub import create_repo, upload_folder

from difftuner.hparams import ModelArguments, DataArguments, DiffusionTrainingArguemnts, FinetuningArguments

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


class CustomFullTrainer:
    def __init__(self, 
                 model_args: ModelArguments = None, 
                 training_args: DiffusionTrainingArguemnts = None, 
                 finetuning_args: FinetuningArguments = None, 
                 train_dataloader: DataLoader = None, 
                 tokenizer: PreTrainedTokenizer = None,
                 accelerator: Accelerator = None, 
                 noise_scheduler: DDPMScheduler = None, 
                 unet: UNet2DConditionModel = None, 
                 controlnet: ControlNetModel = None, 
                 vae: AutoencoderKL = None, 
                 text_encoder: CLIPTextModel = None):
        self.model_args = model_args
        self.training_args = training_args
        self.finetuning_args = finetuning_args

        self.train_dataloader = train_dataloader
        self.tokenizer = tokenizer

        self.accelerator = accelerator
        self.noise_scheduler = noise_scheduler
        self.unet = unet
        self.controlnet = controlnet
        self.vae = vae
        self.text_encoder = text_encoder

    def train(self):
        model_args = self.model_args
        training_args = self.training_args
        finetuning_args = self.finetuning_args
        train_dataloader = self.train_dataloader

        accelerator = self.accelerator
        noise_scheduler = self.noise_scheduler
        unet = self.unet
        vae = self.vae
        text_encoder = self.text_encoder

        # Create EMA for the unet.
        if training_args.use_ema:
            ema_unet = UNet2DConditionModel.from_pretrained(
                model_args.model_name_or_path, subfolder="unet", revision=model_args.revision
            )
            ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    if training_args.use_ema:
                        ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                    for i, model in enumerate(models):
                        model.save_pretrained(os.path.join(output_dir, "unet"))

                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()

            def load_model_hook(models, input_dir):
                if training_args.use_ema:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                    ema_unet.load_state_dict(load_model.state_dict())
                    ema_unet.to(accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        
        # Move vae and text_encoder to device and cast to weight_dtype
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        if training_args.scale_lr:
            training_args.learning_rate = (
                training_args.learning_rate * training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size  * accelerator.num_processes
            )

        # Initialize the optimizer
        if training_args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        # init optimizer
        optimizer = optimizer_cls(
            unet.parameters(),
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            weight_decay=training_args.weight_decay,
            eps=training_args.adam_epsilon,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)

        if training_args.max_steps <= 0:
            training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(training_args.max_steps * accelerator.num_processes),
            num_training_steps=training_args.max_steps * accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        unet_lora_parameters, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet_lora_parameters, optimizer, train_dataloader, lr_scheduler
        )

        if training_args.use_ema:
            ema_unet.to(accelerator.device)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / training_args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            training_args.max_steps = int(training_args.num_train_epochs) * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            tracker_config = dict(vars(training_args))
            accelerator.init_trackers(training_args.tracker_project_name, tracker_config)

        # Training!
        total_batch_size = training_args.per_device_train_batch_size  * accelerator.num_processes * training_args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size }")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {training_args.max_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if training_args.resume_from_checkpoint:
            if training_args.resume_from_checkpoint != "latest":
                path = os.path.basename(training_args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(training_args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{training_args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                training_args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(training_args.output_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch

        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, training_args.max_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        for epoch in range(first_epoch, training_args.num_train_epochs):
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)

                    # Optional
                    if training_args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += training_args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                    if training_args.input_perturbation:
                        new_noise = noise + training_args.input_perturbation * torch.randn_like(noise)

                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if training_args.input_perturbation:
                        noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                    else:
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Get the target for loss depending on the prediction type
                    if training_args.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        noise_scheduler.register_to_config(prediction_type=training_args.prediction_type)

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    # Predict the noise residual and compute loss
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    
                    if training_args.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        if noise_scheduler.config.prediction_type == "v_prediction":
                            # Velocity objective requires that we add one to SNR values before we divide by them.
                            snr = snr + 1
                        mse_loss_weights = (
                            torch.stack([snr, training_args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        )

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(training_args.per_device_train_batch_size )).mean()
                    train_loss += avg_loss.item() / training_args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = unet_lora_parameters
                        accelerator.clip_grad_norm_(params_to_clip, training_args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if training_args.use_ema:
                        ema_unet.step(unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % training_args.save_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if training_args.save_total_limit is not None:
                                checkpoints = os.listdir(training_args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= training_args.save_total_limit:
                                    num_to_remove = len(checkpoints) - training_args.save_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(training_args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= training_args.max_steps:
                    break
            
            if accelerator.is_main_process:
                if training_args.validation_prompt is not None and epoch % training_args.validation_epochs == 0:
                    if training_args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    logger.info(
                        f"Running validation... \n Generating {training_args.num_validation_images} images with prompt:"
                        f" {training_args.validation_prompt}."
                    )

                    self.validation()

                    if training_args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())

        # Save the lora layers
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(unet)
            if training_args.use_ema:
                ema_unet.copy_to(unet.parameters())
            
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_args.model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=model_args.revision
            )
            pipeline.save_pretrained(training_args.output_dir)
            
        accelerator.end_training()
        
    def predict(self):
        self.validation(True)
    
    def validation(self, is_test=False):
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        if is_test:
            pipeline = DiffusionPipeline.from_pretrained(
                self.training_args.output_dir,
                safety_checker=None
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_args.model_name_or_path,
                vae=self.accelerator.unwrap_model(self.vae),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                tokenizer=self.tokenizer,
                unet=self.accelerator.unwrap_model(self.unet),
                safety_checker=None,
                revision=self.model_args.revision,
                variant=self.model_args.variant,
                torch_dtype=weight_dtype,
            )
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if self.training_args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        if self.training_args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.accelerator.device).manual_seed(self.training_args.seed)

        images = []
        for i in range(self.training_args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(self.training_args.validation_prompt, num_inference_steps=30, generator=generator).images[0]

            images.append(image)

        for tracker in self.accelerator.trackers:
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "test" if is_test else "validation": [
                            wandb.Image(image, caption=f"{i}: {self.training_args.validation_prompt}")
                            for i, image in enumerate(images)
                        ]
                    }
                )
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        torch.cuda.empty_cache()