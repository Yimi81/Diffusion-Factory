from typing import Literal, Optional
from dataclasses import asdict, dataclass, field

from transformers.training_args import TrainingArguments
from transformers.utils import add_start_docstrings
from transformers.generation.configuration_utils import GenerationConfig

@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class DiffusionTrainingArguemnts(TrainingArguments):
    mixed_precision: Optional[str] = field(
        default="no",
        metadata={"help": "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                          " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                          " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.",
                  "choices": ["no", "fp16", "bf16"]}
    ),
    prior_generation_precision: Optional[str] = field(
        default=None,
        metadata={"help": "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                          " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32.",
                  "choices": ["no", "fp32", "fp16", "bf16"]}
    )
    use_ema: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use EMA model."}
    )
    scale_lr: Optional[bool] = field(
        default=False,
        metadata={"help": "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size."}
    )
    use_8bit_adam: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use 8-bit Adam from bitsandbytes."}
    )
    tracker_project_name: Optional[str] = field(
        default="text2image-fine-tune",
        metadata={"help": "The `project_name` argument passed to Accelerator.init_trackers for"
                          " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"}
    )
    noise_offset: Optional[float] = field(
        default=0,
        metadata={"help": "The scale of noise offset."}
    )
    input_perturbation: Optional[float] = field(
        default=None,
        metadata={"help": "The scale of input perturbation. Recommended 0.1."}
    )
    prediction_type: Optional[str] = field(
        default=None,
        metadata={"help": "The prediction type that the model was trained on.  Choose between 'epsilon' or 'v_prediction' or leave `None`. "
                          "If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen."
                          "Use 'epsilon' for Stable Diffusion v1.X and Stable Diffusion v2 Base. Use 'v_prediction' for Stable Diffusion v2."}
    )
    snr_gamma: Optional[float] = field(
        default=None,
        metadata={"help": "SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0."
                          "More details here: https://arxiv.org/abs/2303.09556."}
    )
    validation_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "A prompt that is sampled during training for inference."}
    )
    validation_image: Optional[str] = field(
        default=None,
        metadata={"help": "A image that is sampled during training for inference."}
    )
    num_validation_images: Optional[int] = field(
        default=4,
        metadata={"help": "Number of images that should be generated during validation with `validation_prompt`."}
    )
    validation_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
                          "`args.validation_prompt` multiple times: `args.num_validation_images`."}
    )
    train_text_encoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train the text encoder. If set, the text encoder should be float32 precision."}
    )
    pre_compute_text_embeddings: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`."}
    )
    text_encoder_use_attention_mask: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use attention mask for the text encoder"}
    )
    with_prior_preservation: Optional[bool] = field(
        default=False,
        metadata={"help": "Flag to add prior preservation loss."}
    )
    class_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The prompt to specify images in the same class as provided instance images."}
    )
    instance_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The prompt with identifier specifying the instance."}
    )
    class_labels_conditioning: Optional[str] = field(
        default=None,
        metadata={"help": "The optional `class_label` conditioning to pass to the unet, available values are `timesteps`."}
    )
    prior_loss_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "The weight of prior preservation loss."}
    )
    num_class_images: Optional[int] = field(
        default=100,
        metadata={"help": "Minimal class images for prior preservation loss. If there are not enough images already present in."
                          " class_data_dir, additional images will be sampled with class_prompt."}
    ),
    sample_batch_size: Optional[int] = field(
        default=4,
        metadata={"help": "Batch size (per device) for sampling images."}
    )
    tokenizer_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the tokenizer. If not set, will default to the tokenizer's max length."}
    )
    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values and `GenerationConfig` by dictionaries (for JSON
        serialization support). It obfuscates the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = super().to_dict()
        for k, v in d.items():
            if isinstance(v, GenerationConfig):
                d[k] = v.to_dict()
        return d
