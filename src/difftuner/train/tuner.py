from typing import TYPE_CHECKING, Any, Dict, List, Optional

from difftuner.extras.logging import get_logger
from difftuner.model import get_train_args
from difftuner.train.full import run_full
from difftuner.train.lora import run_lora
from difftuner.train.controlnet import run_controlnet

import accelerate
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed


logger = get_logger(__name__)

def run_exp(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, training_args, finetuning_args = get_train_args(args)
    logger.warning(f"finetuning_type: {finetuning_args.finetuning_type}")

    accelerator_project_config = ProjectConfiguration(project_dir=training_args.output_dir, logging_dir=training_args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=training_args.mixed_precision,
        log_with=training_args.report_to,
        project_config=accelerator_project_config,
    )
    logger.info(accelerator.state)

    if finetuning_args.finetuning_type == "full":
        run_full(model_args, data_args, training_args, finetuning_args, accelerator)
    elif finetuning_args.finetuning_type == "lora":
        run_lora(model_args, data_args, training_args, finetuning_args, accelerator)
    elif finetuning_args.finetuning_type == "controlnet":
        run_controlnet(model_args, data_args, training_args, finetuning_args, accelerator)
    else:
        raise ValueError("Unknown task.")