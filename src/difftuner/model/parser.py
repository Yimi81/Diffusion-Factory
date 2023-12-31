import torch
import logging
import transformers
import datasets
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

    # Setup logging
    if training_args.should_log:
        log_level = training_args.get_process_log_level()
        _set_transformers_logging(log_level)

    if finetuning_args.finetuning_type != "dreambooth" and data_args.dataset_name is None and data_args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if data_args.dataset_name is not None and data_args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if training_args.validation_prompt is None and training_args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")
    
    if finetuning_args.finetuning_type == "dreambooth" and data_args.dreambooth_data_dir is None:
        raise ValueError("`--instance_data_dir` must be set if `--finetuning_type` is dreambooth")

    if training_args.with_prior_preservation:
        if data_args.dreambooth_class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if training_args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if data_args.dreambooth_class_data_dir is not None:
            logger.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if training_args.class_prompt is not None:
            logger.warn("You need not use --class_prompt without --with_prior_preservation.")

    if training_args.train_text_encoder and training_args.pre_compute_text_embeddings:
        raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")
    
    model_args.compute_dtype = (
        torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)
    
    return model_args, data_args, training_args, finetuning_args


def _set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()