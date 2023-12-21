import os
import itertools
import math
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path
import numpy as np

import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import shutil

from packaging import version
from tqdm.auto import tqdm

from transformers import PreTrainedTokenizer, CLIPTextModel, CLIPTokenizer, AutoTokenizer
from torchvision import transforms

from difftuner.data import get_dataset, preprocess_dataset, collate_fn
from difftuner.model import load_scheduler_and_model_and_tokenizer
from difftuner.extras.logging import get_logger

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, ControlNetModel, DiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.models.lora import LoRALinearLayer
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)

from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import unet_lora_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose

import accelerate
from accelerate import Accelerator

from huggingface_hub import create_repo, upload_folder

from difftuner.hparams import ModelArguments, DataArguments, DiffusionTrainingArguemnts, FinetuningArguments

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


# TODO: This function should be removed once training scripts are rewritten in PEFT
def text_encoder_lora_state_dict(text_encoder):
    state_dict = {}

    def text_encoder_attn_modules(text_encoder):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        attn_modules = []

        if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))

        return attn_modules

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


class CustomDreamBoothTrainer:
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

        # now we will add new LoRA weights to the attention layers
        # It's important to realize here how many attention weights will be added and of which sizes
        # The sizes of the attention layers consist only of two different variables:
        # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
        # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

        # Let's first see how many attention processors we will have to set.
        # For Stable Diffusion, it should be equal to:
        # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
        # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
        # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
        # => 32 layers
        # Set correct lora layers
        unet_lora_parameters = []
        for attn_processor_name, attn_processor in unet.attn_processors.items():
            # Parse the attention module.
            attn_module = unet
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)

            # Set the `lora_layer` attribute of the attention-related matrices.
            attn_module.to_q.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_q.in_features, out_features=attn_module.to_q.out_features, rank=finetuning_args.lora_rank
                )
            )
            attn_module.to_k.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_k.in_features, out_features=attn_module.to_k.out_features, rank=finetuning_args.lora_rank
                )
            )

            attn_module.to_v.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_v.in_features, out_features=attn_module.to_v.out_features, rank=finetuning_args.lora_rank
                )
            )
            attn_module.to_out[0].set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_out[0].in_features,
                    out_features=attn_module.to_out[0].out_features,
                    rank=finetuning_args.lora_rank,
                )
            )

            # Accumulate the LoRA params to optimize.
            unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

            if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
                attn_module.add_k_proj.set_lora_layer(
                    LoRALinearLayer(
                        in_features=attn_module.add_k_proj.in_features,
                        out_features=attn_module.add_k_proj.out_features,
                        rank=finetuning_args.rank,
                    )
                )
                attn_module.add_v_proj.set_lora_layer(
                    LoRALinearLayer(
                        in_features=attn_module.add_v_proj.in_features,
                        out_features=attn_module.add_v_proj.out_features,
                        rank=finetuning_args.rank,
                    )
                )
                unet_lora_parameters.extend(attn_module.add_k_proj.lora_layer.parameters())
                unet_lora_parameters.extend(attn_module.add_v_proj.lora_layer.parameters())
        
        
        # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
        # So, instead, we monkey-patch the forward calls of its attention-blocks.
        if training_args.train_text_encoder:
            # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
            text_lora_parameters = LoraLoaderMixin._modify_text_encoder(text_encoder, dtype=torch.float32, rank=finetuning_args.rank)

        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    # there are only two options here. Either are just the unet attn processor layers
                    # or there are the unet and text encoder atten layers
                    unet_lora_layers_to_save = None
                    text_encoder_lora_layers_to_save = None

                    for model in models:
                        if isinstance(model, type(accelerator.unwrap_model(unet))):
                            unet_lora_layers_to_save = unet_lora_state_dict(model)
                        elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                            text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(model)
                        else:
                            raise ValueError(f"unexpected save model: {model.__class__}")

                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()

                    LoraLoaderMixin.save_lora_weights(
                        output_dir,
                        unet_lora_layers=unet_lora_layers_to_save,
                        text_encoder_lora_layers=text_encoder_lora_layers_to_save,
                    )

            def load_model_hook(models, input_dir):
                unet_ = None
                text_encoder_ = None

                while len(models) > 0:
                    model = models.pop()

                    if isinstance(model, type(accelerator.unwrap_model(unet))):
                        unet_ = model
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                        text_encoder_ = model
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
                LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)
                LoraLoaderMixin.load_lora_into_text_encoder(
                    lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_
                )

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        
        # Move unet, vae and text_encoder to device and cast to weight_dtype
        unet.to(accelerator.device, dtype=weight_dtype)
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

        # Optimizer creation
        params_to_optimize = (
            itertools.chain(unet_lora_parameters, text_lora_parameters)
            if training_args.train_text_encoder
            else unet_lora_parameters
        )

        # init optimizer
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
        unet_lora_parameters, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet_lora_parameters, optimizer, train_dataloader, lr_scheduler
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
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
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
                    logger.info(
                        f"Running validation... \n Generating {training_args.num_validation_images} images with prompt:"
                        f" {training_args.validation_prompt}."
                    )

                    self.validation()

        # Save the lora layers
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unet = unet.to(torch.float32)
            unet.save_attn_procs(training_args.output_dir)
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
                self.model_args.pretrained_model_name_or_path, revision=self.model_args.revision, variant=self.model_args.variant, torch_dtype=weight_dtype
            )
            # load attention processors
            pipeline.unet.load_attn_procs(self.training_args.output_dir)
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                self.model_args.pretrained_model_name_or_path,
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