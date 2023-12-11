import os
import math
import torch
from packaging import version
from types import MethodType
from typing import TYPE_CHECKING, Literal, Optional, Tuple

from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer

from transformers.utils.versions import require_version


from difftuner.extras.logging import reset_logging, get_logger
from difftuner.hparams import FinetuningArguments
from difftuner.hparams import DiffusionTrainingArguemnts

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, ControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available

if TYPE_CHECKING:
    from difftuner.hparams import ModelArguments


logger = get_logger(__name__)


require_version("datasets>=2.14.0", "To fix: pip install datasets>=2.14.0")


def load_scheduler_and_model_and_tokenizer(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    training_args: "DiffusionTrainingArguemnts",
    is_trainable: Optional[bool] = False,
) -> Tuple[DDPMScheduler, UNet2DConditionModel, ControlNetModel | None, AutoencoderKL, CLIPTextModel, CLIPTokenizer]:
    r"""
    Loads scheduler, model and tokenizer.

    Support both training and inference.
    """

    logger.info(f"{'-'*20} Loading scheduler, model and tokenizer {'-'*20}")

    noise_scheduler = DDPMScheduler.from_pretrained(model_args.model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_args.model_name_or_path, subfolder="tokenizer", revision=model_args.revision
    )

    unet = UNet2DConditionModel.from_pretrained(
        model_args.model_name_or_path, subfolder="unet", revision=model_args.non_ema_revision
    )
    controlnet = None
    text_encoder = CLIPTextModel.from_pretrained(
        model_args.model_name_or_path, subfolder="text_encoder", revision=model_args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        model_args.model_name_or_path, subfolder="vae", revision=model_args.revision
    )

    if finetuning_args.finetuning_type == "full":
        # Freeze vae and text_encoder and set unet to trainable
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.train()

        if training_args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

    elif finetuning_args.finetuning_type == "lora":
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

    elif finetuning_args.finetuning_type == "controlnet":
        if model_args.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            controlnet = ControlNetModel.from_pretrained(model_args.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from unet")
            controlnet = ControlNetModel.from_unet(unet)

        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)
        controlnet.train()
        
        if training_args.gradient_checkpointing:
            controlnet.enable_gradient_checkpointing()
    
    elif finetuning_args.finetuning_type == "dreambooth":
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        
        if training_args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if training_args.train_text_encoder:
                text_encoder.gradient_checkpointing_enable()

    if model_args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            if controlnet:
                controlnet.enable_xformers_memory_efficient_attention()

        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    

    
    return noise_scheduler, unet, controlnet, vae, text_encoder, tokenizer