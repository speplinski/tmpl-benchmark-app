# The Most Polish Landscape (Semantic Image Synthesis)

This application evaluates the performance of hardware configurations (GPUs) for the installation *The Most Polish Landscape*. By benchmarking the processing times for tasks like inference, image transformation, and saving, it ensures the selected hardware meets the installation's performance requirements.

---

## Installation

Clone this repository and install the required dependencies (Python 3+ and PyTorch 1.0):

```bash
git clone https://github.com/speplinski/tmpl-benchmark-app.git
cd tmpl-benchmark-app/
pip install -r requirements.txt
```

To reproduce the results, you would need a machine with two NVIDIA GPUs (e.g., A100), configured to support multi-GPU training.

## Dataset Preparation

For TMPL, the datasets must be downloaded beforehand. A script has been provided to simplify the download process.

**Preparing TMPL Dataset**. The dataset can be downloaded To download and prepare the dataset, navigate to the `datasets` directory and run the provided script:

```bash
cd datasets/
./download_dataset.sh
cd ..
```

This script will automatically download and extract the required dataset files and organize them in the appropriate structure under the `datasets/` directory.

## Generating Images Using Pretrained Model

Once the dataset is ready, the result images can be generated using pretrained models.

1. **Download Pretrained Models**

    Navigate to the `checkpoints` directory and run the provided script to download and extract the pretrained models. This script will automatically download the pretrained models archive and extract its contents into the `checkpoints/` directory.

    ```bash
    cd checkpoints
    ./download_ckpts.sh
    cd ..
    ```

2. **Generate images**

    Use the pretrained model to generate images.
    
    ```bash
    python3 gen.py \
      --batchSize 4 \
      --gpu_ids 0,1 \
      --which_epoch 80
    ```

    **Note for Single-GPU Systems**
    If there is only one GPU, set `--gpu_ids 0` and adjust `--batchSize` to fit memory constraints. The application should generate at least 4 images per second to meet the performance requirements.

3. **Output Location**

    The generated images are stored in `./results/` by default.

    ![Benchmark](assets/benchmark.png?raw=true)

    **Performance Benchmark**
    The benchmark above serves as a reference for performance. The processing times shown (inference, transformation, and saving) should not be significantly slower on comparable hardware.

## Code Structure

- `gen.py`: the entry point for generating.
- `trainers/pix2pix_trainer.py`: harnesses and reports the progress of training.
- `models/pix2pix_model.py`: creates the networks, and compute the losses
- `models/networks/`: defines the architecture of all models
- `options/`: creates option lists using `argparse` package. More individuals are dynamically added in other files as well. Please see the section below.
- `data/`: defines the class for loading images and label maps.

## Options

This repository contains many configurable options. Some are model-specific, while others have different default values depending on the dataset or network in use.

To dynamically adjust options based on the dataset or model, the `BaseOption` class calls the static method `modify_commandline_options`. This modifies the argparse parser with appropriate settings.

## Acknowledgments

This code borrows heavily from `pix2pixHD` and `SPADE`. Special thanks to:

- Jiayuan Mao for his Synchronized Batch Normalization code.
- Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu for Semantic Image Synthesis with Spatially-Adaptive Normalization.
