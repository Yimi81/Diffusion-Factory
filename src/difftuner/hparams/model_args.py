from typing import Any, Dict, Literal, Optional
from dataclasses import asdict, dataclass, field


@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    controlnet_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained controlnet model or model identifier from huggingface.co/models. If not specified controlnet weights are initialized from unet."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory where the downloaded models and datasets will be stored.."}
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision of pretrained model identifier from huggingface.co/models."}
    )
    variant: Optional[str] = field(
        default=None,
        metadata={"help": "Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16"}
    )
    non_ema_revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
                          " remote repository specified with --model_name_or_path."}
    )
    enable_xformers_memory_efficient_attention: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use xformers."}
    )


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)