from typing import Any, Dict, Optional

from difftuner.extras.logging import get_logger
from difftuner.model import get_train_args
from difftuner.train.full import run_full
from difftuner.train.lora import run_lora, run_sdxl_lora
from difftuner.train.controlnet import run_controlnet
from difftuner.train.dreambooth import run_dreambooth

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs


logger = get_logger(__name__)

def run_exp(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, training_args, finetuning_args = get_train_args(args)
    logger.warning(f"finetuning_type: {finetuning_args.finetuning_type}")

    accelerator_project_config = ProjectConfiguration(project_dir=training_args.output_dir, logging_dir=training_args.logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=training_args.mixed_precision,
        log_with=training_args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    logger.info(accelerator.state)

    if finetuning_args.finetuning_type == "full":
        run_full(model_args, data_args, training_args, finetuning_args, accelerator)
    elif finetuning_args.finetuning_type == "lora":
        run_lora(model_args, data_args, training_args, finetuning_args, accelerator)
    elif finetuning_args.finetuning_type == "dreambooth":
        run_dreambooth(model_args, data_args, training_args, finetuning_args, accelerator)
    elif finetuning_args.finetuning_type == "controlnet":
        run_controlnet(model_args, data_args, training_args, finetuning_args, accelerator)
    elif finetuning_args.finetuning_type == "sdxl-lora":
        run_sdxl_lora(model_args, data_args, training_args, finetuning_args, accelerator)
    else:
        raise ValueError("Unknown task.")