import os
import torch
import datasets
import transformers
import diffusers
from typing import Any, Dict, Optional, Tuple
from transformers import HfArgumentParser

from difftuner.extras.logging import get_logger
from difftuner.extras.misc import parse_args

from difftuner.hparams import (
    ModelArguments,
    DataArguments,
    DiffusionTrainingArguemnts,
    FinetuningArguments
)
logger = get_logger(__name__)

_TRAIN_ARGS = [
    ModelArguments, DataArguments, DiffusionTrainingArguemnts, FinetuningArguments
]
_TRAIN_CLS = Tuple[
    ModelArguments, DataArguments, DiffusionTrainingArguemnts, FinetuningArguments
]

def parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    return parse_args(parser, args)

def get_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    model_args, data_args, training_args, finetuning_args = parse_train_args(args)

    if data_args.dataset_name is None and data_args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if data_args.dataset_name is not None and data_args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if training_args.validation_prompt is None and training_args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")
    
    # Setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    diffusers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    model_args.compute_dtype = (
        torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)
    
    return model_args, data_args, training_args, finetuning_args