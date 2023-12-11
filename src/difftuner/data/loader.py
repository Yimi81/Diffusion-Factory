import os
from typing import TYPE_CHECKING, Any, Dict, List, Union

import torch

from datasets import load_dataset

from difftuner.extras.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from difftuner.hparams import ModelArguments, DataArguments


logger = get_logger(__name__)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments"
) -> Union["Dataset", "IterableDataset"]:
    max_samples = data_args.max_samples

    logger.info(f"{'-'*20} Loading dataset {'-'*20}")
    
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            data_dir=data_args.train_data_dir
        )
    else:
        data_files = {}
        if data_args.train_data_dir is not None:
            data_files["train"] = os.path.join(data_args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir
        )
    
    if max_samples is not None: # truncate dataset
        dataset = dataset.select(range(min(len(dataset), max_samples)))
        
    return dataset