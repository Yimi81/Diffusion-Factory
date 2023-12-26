import random
import torch 
from torch.utils.data import Dataset as DT

import gc

from typing import TYPE_CHECKING, Union

import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import crop
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose

from difftuner.extras.logging import get_logger
from difftuner.train.utils import tokenize_prompt


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers.tokenization_utils import PreTrainedTokenizer

    from difftuner.hparams import (
        DataArguments,
        DiffusionTrainingArguemnts,
        FinetuningArguments
    )


logger = get_logger(__name__)


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


class DreamBoothDataset(DT):
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
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

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
            # logger.info(f"class_data_root: {self.class_data_root}")
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
    

def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "DiffusionTrainingArguemnts",
    finetuning_args: "FinetuningArguments", 
):
    
    logger.info(f"Preprocess dataset")

    if finetuning_args.finetuning_type == "dreambooth":
        if training_args.pre_compute_text_embeddings:

            def compute_text_embeddings(prompt):
                with torch.no_grad():
                    text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=training_args.tokenizer_max_length)
                    prompt_embeds = encode_prompt(
                        text_encoder,
                        text_inputs.input_ids,
                        text_inputs.attention_mask,
                        text_encoder_use_attention_mask=training_args.text_encoder_use_attention_mask,
                    )

                return prompt_embeds

            pre_computed_encoder_hidden_states = compute_text_embeddings(training_args.instance_prompt)

            if training_args.class_prompt is not None:
                pre_computed_class_prompt_encoder_hidden_states = compute_text_embeddings(training_args.class_prompt)
            else:
                pre_computed_class_prompt_encoder_hidden_states = None

            text_encoder = None
            tokenizer = None

            gc.collect()
            torch.cuda.empty_cache()
        else:
            pre_computed_encoder_hidden_states = None
            pre_computed_class_prompt_encoder_hidden_states = None

        train_dataset = DreamBoothDataset(
            instance_data_root=data_args.dreambooth_data_dir,
            instance_prompt=training_args.instance_prompt,
            class_data_root=data_args.dreambooth_class_data_dir if training_args.with_prior_preservation else None,
            class_prompt=training_args.class_prompt,
            class_num=training_args.num_class_images,
            tokenizer=tokenizer,
            size=data_args.resolution,
            center_crop=data_args.center_crop,
            encoder_hidden_states=pre_computed_encoder_hidden_states,
            class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
            tokenizer_max_length=training_args.tokenizer_max_length,
        )
    else:
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
    
    logger.info(f"Finish preprocess dataset")

    return train_dataset



def sdxl_preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer_one: "PreTrainedTokenizer",
    tokenizer_two: "PreTrainedTokenizer",
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

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        tokens_one = tokenize_prompt(tokenizer_one, captions)
        tokens_two = tokenize_prompt(tokenizer_two, captions)
        return tokens_one, tokens_two
    
    # Preprocessing the datasets.
    train_resize = transforms.Resize(data_args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(data_args.resolution) if data_args.center_crop else transforms.RandomCrop(data_args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if data_args.center_crop:
                y1 = max(0, int(round((image.height - data_args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - data_args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (data_args.resolution, data_args.resolution))
                image = crop(image, y1, x1, h, w)
            if data_args.random_flip and random.random() < 0.5:
                # flip
                x1 = image.width - x1
                image = train_flip(image)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        tokens_one, tokens_two = tokenize_captions(examples)
        examples["input_ids_one"] = tokens_one
        examples["input_ids_two"] = tokens_two
        return examples
    
    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        if data_args.max_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)
    
    return train_dataset