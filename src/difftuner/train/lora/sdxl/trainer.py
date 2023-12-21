import os
import math
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import shutil
from tqdm.auto import tqdm

from transformers import PreTrainedTokenizer, CLIPPreTrainedModel

from difftuner.extras.logging import get_logger

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import is_wandb_available
from diffusers.loaders import LoraLoaderMixin

from accelerate import Accelerator

from difftuner.hparams import ModelArguments, DiffusionTrainingArguemnts, FinetuningArguments, DataArguments
from difftuner.train.utils import encode_prompt

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


class SDXLCustomLoraTrainer:
    def __init__(self, 
                 data_args: DataArguments, 
                 model_args: ModelArguments, 
                 training_args: DiffusionTrainingArguemnts, 
                 finetuning_args: FinetuningArguments, 
                 train_dataloader: DataLoader, 
                 tokenizer_one: PreTrainedTokenizer,
                 tokenizer_two: PreTrainedTokenizer,
                 accelerator: Accelerator, 
                 noise_scheduler: DDPMScheduler, 
                 unet: UNet2DConditionModel, 
                 vae: AutoencoderKL, 
                 text_encoder_one: CLIPPreTrainedModel,
                 text_encoder_two: CLIPPreTrainedModel):
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        self.finetuning_args = finetuning_args
        
        self.train_dataloader = train_dataloader
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two

        self.accelerator = accelerator
        self.noise_scheduler = noise_scheduler
        self.unet = unet
        self.vae = vae
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two

    def train(self):
        data_args = self.data_args
        model_args = self.model_args
        training_args = self.training_args
        finetuning_args = self.finetuning_args
        train_dataloader = self.train_dataloader

        accelerator = self.accelerator
        noise_scheduler = self.noise_scheduler
        unet = self.unet
        vae = self.vae
        text_encoder_one = self.text_encoder_one
        text_encoder_two = self.text_encoder_two


        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        # The VAE is in float32 to avoid NaN losses.
        unet.to(accelerator.device, dtype=weight_dtype)
        if model_args.pretrained_vae_model_name_or_path is None:
            vae.to(accelerator.device, dtype=torch.float32)
        else:
            vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

        # now we will add new LoRA weights to the attention layers
        # Set correct lora layers
        unet_lora_config = LoraConfig(
            r=finetuning_args.lora_rank, 
            lora_alpha=finetuning_args.lora_alpha, 
            lora_dropout=finetuning_args.lora_dropout, 
            init_lora_weights="gaussian", 
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        )
        unet.add_adapter(unet_lora_config)

        # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
        if training_args.train_text_encoder:
            # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
            text_lora_config = LoraConfig(
                r=finetuning_args.lora_rank, init_lora_weights="gaussian", target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
            )
            text_encoder_one.add_adapter(text_lora_config)
            text_encoder_two.add_adapter(text_lora_config)

        if training_args.mixed_precision == "fp16":
            models = [unet]
            if training_args.train_text_encoder:
                models.extend([text_encoder_one, text_encoder_two])
            for model in models:
                for param in model.parameters():
                    # only upcast trainable parameters (LoRA) into fp32
                    if param.requires_grad:
                        param.data = param.to(torch.float32)

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                # there are only two options here. Either are just the unet attn processor layers
                # or there are the unet and text encoder atten layers
                unet_lora_layers_to_save = None
                text_encoder_one_lora_layers_to_save = None
                text_encoder_two_lora_layers_to_save = None

                for model in models:
                    if isinstance(model, type(accelerator.unwrap_model(unet))):
                        unet_lora_layers_to_save = get_peft_model_state_dict(model)
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                        text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                        text_encoder_two_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

                StableDiffusionXLPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                    text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            unet_ = None
            text_encoder_one_ = None
            text_encoder_two_ = None

            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_ = model
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                    text_encoder_one_ = model
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                    text_encoder_two_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
            LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)

            text_encoder_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder." in k}
            LoraLoaderMixin.load_lora_into_text_encoder(
                text_encoder_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_one_
            )

            text_encoder_2_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k}
            LoraLoaderMixin.load_lora_into_text_encoder(
                text_encoder_2_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_two_
            )

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

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
        params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
        if training_args.train_text_encoder:
            params_to_optimize = (
                params_to_optimize
                + list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
                + list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
            )
        optimizer = optimizer_cls(
            params_to_optimize,
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
        if training_args.train_text_encoder:
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

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
            unet.train()
            if training_args.train_text_encoder:
                text_encoder_one.train()
                text_encoder_two.train()
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    if model_args.pretrained_vae_model_name_or_path is not None:
                        pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                    else:
                        pixel_values = batch["pixel_values"]

                    # Convert images to latent space
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
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

                    # time ids
                    def compute_time_ids(original_size, crops_coords_top_left):
                        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                        target_size = (data_args.resolution, data_args.resolution)
                        add_time_ids = list(original_size + crops_coords_top_left + target_size)
                        add_time_ids = torch.tensor([add_time_ids])
                        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                        return add_time_ids

                    add_time_ids = torch.cat(
                        [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                    )

                    # Get the time_ids and text embedding for conditioning
                    unet_added_conditions = {"time_ids": add_time_ids}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[batch["input_ids_one"], batch["input_ids_two"]],
                    )
                    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

                    # Predict the noise residual and compute loss
                    model_pred = unet(
                        noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions
                    ).sample

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
                        accelerator.clip_grad_norm_(params_to_optimize, training_args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
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

                            unwrapped_unet = accelerator.unwrap_model(unet)
                            unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet)

                            if training_args.train_text_encoder:
                                text_encoder_one = accelerator.unwrap_model(text_encoder_one)
                                text_encoder_two = accelerator.unwrap_model(text_encoder_two)

                                text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one)
                                text_encoder_2_lora_layers = get_peft_model_state_dict(text_encoder_two)
                            else:
                                text_encoder_lora_layers = None
                                text_encoder_2_lora_layers = None

                            StableDiffusionXLPipeline.save_lora_weights(
                                save_directory=training_args.output_dir,
                                unet_lora_layers=unet_lora_state_dict,
                                text_encoder_lora_layers=text_encoder_lora_layers,
                                text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                            )

                            logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= training_args.max_steps:
                    break
            
            if accelerator.is_main_process:
                if training_args.validation_prompt is not None and epoch % training_args.validation_epochs == 0:
                    logger.info(
                        f"Running validation... \n Generating {training_args.num_validation_images} images with prompt:"
                        f" {training_args.validation_prompt}."
                    )

                    self.validation()

        # Save the lora layers
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_unet = accelerator.unwrap_model(unet)
            unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet)

            if training_args.train_text_encoder:
                text_encoder_one = accelerator.unwrap_model(text_encoder_one)
                text_encoder_two = accelerator.unwrap_model(text_encoder_two)

                text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one)
                text_encoder_2_lora_layers = get_peft_model_state_dict(text_encoder_two)
            else:
                text_encoder_lora_layers = None
                text_encoder_2_lora_layers = None

            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=training_args.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_lora_layers,
                text_encoder_2_lora_layers=text_encoder_2_lora_layers,
            )

            del unet
            del text_encoder_one
            del text_encoder_two
            del text_encoder_lora_layers
            del text_encoder_2_lora_layers
            torch.cuda.empty_cache()

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
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_args.pretrained_model_name_or_path, revision=self.model_args.revision, variant=self.model_args.variant, torch_dtype=weight_dtype
            )
            # load attention processors
            pipeline.load_lora_weights(self.training_args.output_dir)
        else:
            # create pipeline
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.training_args.pretrained_model_name_or_path,
                vae=self.vae,
                text_encoder=self.accelerator.unwrap_model(self.text_encoder_one),
                text_encoder_2=self.accelerator.unwrap_model(self.text_encoder_two),
                unet=self.accelerator.unwrap_model(self.unet),
                revision=self.model_args.revision,
                variant=self.model_args.variant,
                torch_dtype=weight_dtype,
            )
            
        pipeline.set_progress_bar_config(disable=True)
        pipeline = pipeline.to(self.accelerator.device)

        if self.model_args.enable_xformers_memory_efficient_attention:
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

        return images
