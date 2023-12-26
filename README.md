# Diffusion-Factory

Easy-to-use Diffusion fine-tuning framework, based on diffusers and peft library.

## Getting Started

### Data Preparation
You can directly use datasets from the hub or upload your custom dataset to the `data` folder. Please refer to [create_dataset](https://huggingface.co/docs/diffusers/training/create_dataset) for the format of the data.

### Installing the dependencies

```bash
git clone https://github.com/Yimi81/Diffusion-Factory.git
conda create -n diffusion_factory python=3.10 -y
conda activate diffusion_factory
cd Diffusion-Factory
pip install -r requirements.txt
```

### Train on a single GPU

```bash
bash scripts/train_text_to_image_lora.sh
```