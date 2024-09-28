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

This code also requires the Synchronized-BatchNorm-PyTorch rep.
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

To reproduce the results, you would need an NVIDIA machine with A100 GPU.

## Dataset Preparation

For TMPL, the datasets must be downloaded beforehand. Please download them on the respective webpages.

**Preparing TMPL Dataset**. The dataset can be downloaded [here](https://storage.googleapis.com/polish_landscape/dataset/test.tar.gz). In particular, you will need to download test.tar.gz. The images, labels, and instance maps should be arranged in the same directory structure as in `datasets/tmpl/`.

```
cd datasets/tmpl
tar xvf test.tar.gz
cd ../../
```

There are different modes to load images by specifying `--preprocess_mode` along with `--load_size`. `--crop_size`. There are options such as `resize_and_crop`, which resizes the images into square images of side length `load_size` and randomly crops to `crop_size`. `scale_shortside_and_crop` scales the image to have a short side of length `load_size` and crops to `crop_size` x `crop_size` square. To see all modes, please use `python train.py --help` and take a look at `data/base_dataset.py`. By default at the training phase, the images are randomly flipped horizontally. To prevent this use `--no_flip`.

## Generating Images Using Pretrained Model

Once the dataset is ready, the result images can be generated using pretrained models.

1. Download the tar of the pretrained models from the [Google Cloud Storage](https://storage.googleapis.com/polish_landscape/checkpoints/checkpoints.tar.gz) (5.21 GB), save it in 'checkpoints/', and run

    ```
    cd checkpoints
    tar xvf checkpoints.tar.gz
    cd ../
    ```

2. Generate images using the pretrained model.
    ```bash    
    python generate.py \
      --name v27 \
      --dataset_mode custom \
      --label_dir datasets/tmpl/test/masks \
      --image_dir datasets/tmpl/test/images \
      --label_nc 12 \
      --batchSize 1 \
      --gpu_ids 0 \
      --no_instance \
      --aspect_ratio 3.0 \
      --crop_size 576 \
      --load_size 1728 \
      --num_upsampling_layers normal \
      --ngf 256 \
      --which_epoch 25 \
      --contain_dontcare_label \
      --use_vae \
      --preprocess_mode fixed
    ```
    `If you are running on CPU mode, append `--gpu_ids -1`.

3. The outputs images are stored at `./results/` by default.

## Code Structure

- `generate.py`: the entry point for generating.
- `trainers/pix2pix_trainer.py`: harnesses and reports the progress of training.
- `models/pix2pix_model.py`: creates the networks, and compute the losses
- `models/networks/`: defines the architecture of all models
- `options/`: creates option lists using `argparse` package. More individuals are dynamically added in other files as well. Please see the section below.
- `data/`: defines the class for loading images and label maps.

## Options

This code repo contains many options. Some options belong to only one specific model, and some options have different default values depending on other options. To address this, the `BaseOption` class dynamically loads and sets options depending on what model, network, and datasets are used. This is done by calling the static method `modify_commandline_options` of various classes. It takes in the`parser` of `argparse` package and modifies the list of options. For example, since TMPL dataset contains a special label "unknown", it sets `--contain_dontcare_label` automatically at `data/tmpl_dataset.py`. You can take a look at `def gather_options()` of `options/base_options.py`, or `models/network/__init__.py` to get a sense of how this works.

## VAE-Style Training with an Encoder For Style Control and Multi-Modal Outputs

To train our model along with an image encoder to enable multi-modal outputs, please use `--use_vae`. The model will create `netE` in addition to `netG` and `netD` and train with KL-Divergence loss.

## Acknowledgments
This code borrows heavily from pix2pixHD and SPADE. We thank Jiayuan Mao for his Synchronized Batch Normalization code and Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu for Semantic Image Synthesis with Spatially-Adaptive Normalization.