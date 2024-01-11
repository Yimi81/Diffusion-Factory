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

### Contributors
<!-- readme: collaborators,contributors -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/Yimi81">
            <img src="https://avatars.githubusercontent.com/u/66633207?v=4" width="100;" alt="Yimi81"/>
            <br />
            <sub><b>YShow</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: collaborators,contributors -end -->