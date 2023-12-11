import json
from typing import Literal, Optional
from dataclasses import asdict, dataclass, field


@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """
    lora_rank: Optional[int] = field(
        default=4,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )
    lora_alpha: Optional[float] = field(
        default=8,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2.0)."}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )

@dataclass
class FinetuningArguments(LoraArguments):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    finetuning_type: Optional[Literal["full", "lora", "dreambooth", "controlnet"]] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."}
    )
    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."}
    )