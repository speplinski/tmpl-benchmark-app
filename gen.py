import os
import time
import glob
import numpy as np
from PIL import Image
from collections import OrderedDict

import data
import torch
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

from models.networks.sync_batchnorm import DataParallelWithCallback

opt = TestOptions().parse()
dataloader = data.create_dataloader(opt)
model = Pix2PixModel(opt)
if len(opt.gpu_ids) > 1:
    model = DataParallelWithCallback(model, device_ids=opt.gpu_ids)
else:
    model = model.to(f"cuda:{opt.gpu_ids[0]}")
model.eval()

output_dir = opt.results_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = glob.glob(os.path.join(output_dir, '*'))
for f in files:
    os.remove(f)

counter = 1  # Initialize counter for sequential numbering

for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    total_start_time = time.time()

    # Measure time for inference
    torch.cuda.synchronize()
    iter_start_time = time.time()
    generated = model(data_i, mode='inference')
    torch.cuda.synchronize()
    iter_time = time.time() - iter_start_time
    print(f"\nTime for model inference in iteration {i}: {iter_time:.4f} seconds")

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        # Sequential numbering for each image
        image_name = f"{counter}"
        print(f"Processing image... {image_name}")

        # Measure time for transformation
        transform_start_time = time.time()

        # Convert tensor to image array
        image_tensor = generated[b].detach().cpu()  # Move to CPU
        image_array = ((image_tensor.numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)  # Normalize to [0, 255]

        # Convert format (C, H, W) -> (H, W, C)
        if image_array.shape[0] == 3:
            image_array = image_array.transpose(1, 2, 0)

        transform_time = time.time() - transform_start_time
        print(f"Time for transformation to image array: {transform_time:.4f} seconds")

        # Path for output image
        output_path_png = os.path.join(output_dir, f"{image_name}.bmp")

        # Save as file and measure save time
        save_start_time = time.time()
        Image.fromarray(image_array, mode="RGB").save(output_path_png, format="BMP")
        save_time_png = time.time() - save_start_time
        print(f"Time to save image: {save_time_png:.4f} seconds")

        # Increment the counter for the next image
        counter += 1

        # Total time for processing one image
        total_time = time.time() - total_start_time
        print(f"Total time for processing image: {total_time:.4f} seconds")
