import os
import json
from typing import List, Literal, Optional
from dataclasses import dataclass, field


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
                          " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
                          " or to a folder containing files that 🤗 Datasets can understand."}
    )
    train_data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "A folder containing the training data. Folder contents must follow the structure described in"
                          " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
                          " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."}
    )
    resolution: Optional[int] = field(
        default=None,
        metadata={"help": "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."}
    )
    center_crop: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
                          " cropped. The images will be resized to the resolution first before cropping."}
    )
    random_flip: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to randomly flip images horizontally"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The config of the Dataset, leave as None if there's only one config."}
    )
    image_column: Optional[str] = field(
        default="image",
        metadata={"help": "The column of the dataset containing an image."}
    )
    conditioning_image_column: Optional[str] = field(
        default="conditioning_image",
        metadata={"help": "The column of the dataset containing the controlnet conditioning image."}
    )
    caption_column: Optional[str] = field(
        default="text",
        metadata={"help": "The column of the dataset containing a caption or a list of captions."}
    )
