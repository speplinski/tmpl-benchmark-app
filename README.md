# The Most Polish Landscape (Semantic Image Synthesis)

## Installation

Clone this repo.
```bash
git clone https://github.com/speplinski/tmpl.git
cd tmpl/
```

This code requires PyTorch 1.0 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```

To reproduce the results, you would need a machine with two NVIDIA GPUs (e.g., A100), configured to support multi-GPU training.

## Dataset Preparation

For TMPL, the datasets must be downloaded beforehand. A script has been provided to simplify the download process.

**Preparing TMPL Dataset**. The dataset can be downloaded To download and prepare the dataset, navigate to the `datasets` directory and run the provided script:

```bash
cd datasets/
./download_dataset.sh
cd ../
```

This script will automatically download and extract the required dataset files and organize them in the appropriate structure under the `datasets/` directory.

## Generating Images Using Pretrained Model

Once the dataset is ready, the result images can be generated using pretrained models.

1. Navigate to the `checkpoints` directory and run the provided script to download and extract the pretrained models. This script will automatically download the pretrained models archive and extract its contents into the `checkpoints/` directory.

    ```bash
    cd checkpoints
    ./download_ckpts.sh
    cd ../
    ```

2. Generate images using the pretrained model.
    ```bash
    python3 gen.py \
      --batchSize 4 \
      --gpu_ids 0,1 \
      --which_epoch 80
    ```
    If you have only one GPU, set `--gpu_ids 0` and adjust `--batchSize` to fit memory constraints.

3. The outputs images are stored at `./results/` by default.

    ![Benchmark](assets/benchmark.png?raw=true)

    **Note:** The benchmark above serves as a reference for performance. The processing times shown (inference, transformation, and saving) should not be significantly slower on comparable hardware.

## Code Structure

- `gen.py`: the entry point for generating.
- `trainers/pix2pix_trainer.py`: harnesses and reports the progress of training.
- `models/pix2pix_model.py`: creates the networks, and compute the losses
- `models/networks/`: defines the architecture of all models
- `options/`: creates option lists using `argparse` package. More individuals are dynamically added in other files as well. Please see the section below.
- `data/`: defines the class for loading images and label maps.

## Options

This code repo contains many options. Some options belong to only one specific model, and some options have different default values depending on other options. To address this, the `BaseOption` class dynamically loads and sets options depending on what model, network, and datasets are used. This is done by calling the static method `modify_commandline_options` of various classes. It takes in the`parser` of `argparse` package and modifies the list of options. For example, since TMPL dataset contains a special label "unknown", it sets `--contain_dontcare_label` automatically at `data/tmpl_dataset.py`. You can take a look at `def gather_options()` of `options/base_options.py`, or `models/network/` to get a sense of how this works.

## Acknowledgments
This code borrows heavily from pix2pixHD and SPADE. We thank Jiayuan Mao for his Synchronized Batch Normalization code and Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu for Semantic Image Synthesis with Spatially-Adaptive Normalization.