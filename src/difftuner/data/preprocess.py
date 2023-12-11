import os
import random

from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Tuple, Union

import numpy as np
from torchvision import transforms

from difftuner.extras.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers.tokenization_utils import PreTrainedTokenizer

    from difftuner.hparams import (
        DataArguments,
        DiffusionTrainingArguemnts,
        FinetuningArguments
    )


logger = get_logger(__name__)


def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "DiffusionTrainingArguemnts",
    finetuning_args: "FinetuningArguments", 
):
    
    logger.info(f"{'-'*20} Preprocess dataset {'-'*20}")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    if data_args.image_column is None:
        image_column = column_names[0]
    else:
        image_column = data_args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{data_args.image_column}' needs to be one of: {', '.join(column_names)}"
            )

    if data_args.caption_column is None:
        caption_column = column_names[1]
    else:
        caption_column = data_args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{data_args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    if finetuning_args.finetuning_type == "controlnet":
        if data_args.conditioning_image_column is None:
            conditioning_image_column = column_names[2]
            logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
        else:
            conditioning_image_column = data_args.conditioning_image_column
            if conditioning_image_column not in column_names:
                raise ValueError(
                    f"`--conditioning_image_column` value '{data_args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )

        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")     

        return inputs.input_ids
    
    train_transforms = transforms.Compose(
        [
            transforms.Resize(data_args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(data_args.resolution) if data_args.center_crop else transforms.RandomCrop(data_args.resolution),
            transforms.RandomHorizontalFlip() if data_args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(data_args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(data_args.resolution),
            transforms.ToTensor(),
        ]
    )
    
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [train_transforms(image) for image in images]

        if finetuning_args.finetuning_type == "controlnet":
            conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
            conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]
            examples["conditioning_pixel_values"] = conditioning_images

        examples["pixel_values"] = images
        examples["input_ids"] = tokenize_captions(examples)
        return examples
    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        if data_args.max_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)
    
    return train_dataset