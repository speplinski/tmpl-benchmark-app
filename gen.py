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

def process_batch(model, data_i, output_dir, counter):
    """Process a single batch of images"""
    total_start_time = time.time()
    torch.cuda.synchronize()
    iter_start_time = time.time()

    try:
        if isinstance(model, DataParallelWithCallback):
            generated = model.module(data_i, mode='inference')
        else:
            generated = model(data_i, mode='inference')
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return counter

    torch.cuda.synchronize()
    iter_time = time.time() - iter_start_time
    print(f"\nTime for model inference: {iter_time:.4f} seconds")

    for b in range(generated.shape[0]):
        try:
            image_name = f"{counter}"
            print(f"Processing image... {image_name}")

            transform_start_time = time.time()
            image_tensor = generated[b].detach().cpu()
            image_array = ((image_tensor.numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

            if image_array.shape[0] == 3:
                image_array = image_array.transpose(1, 2, 0)

            transform_time = time.time() - transform_start_time
            print(f"Time for transformation to image array: {transform_time:.4f} seconds")

            output_path = os.path.join(output_dir, f"{image_name}.bmp")
            save_start_time = time.time()
            Image.fromarray(image_array, mode="RGB").save(output_path, format="BMP")
            save_time = time.time() - save_start_time
            print(f"Time to save image: {save_time:.4f} seconds")

            counter += 1

            total_time = time.time() - total_start_time
            print(f"Total time for processing image: {total_time:.4f} seconds")
        except Exception as e:
            print(f"Error processing image {counter}: {str(e)}")
            counter += 1
            continue

    return counter

def main():
    opt = TestOptions().parse()

    # Create output directory if it doesn't exist
    output_dir = opt.results_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize model
    model = Pix2PixModel(opt)
    model.eval()

    if len(opt.gpu_ids) > 1:
        model = DataParallelWithCallback(model, device_ids=opt.gpu_ids)
    else:
        model = model.to(f"cuda:{opt.gpu_ids[0]}")

    processed_count = 0
    last_dataloader_size = 0

    while True:
        try:
            # Create new dataloader to check for new files
            dataloader = data.create_dataloader(opt)
            current_dataloader_size = len(dataloader.dataset)

            # If no new files and we've processed everything, wait
            if current_dataloader_size == last_dataloader_size and processed_count >= current_dataloader_size:
                print("Waiting for new data...")
                time.sleep(1)
                continue

            # Process only new files
            for i, data_i in enumerate(dataloader):
                if processed_count >= current_dataloader_size:
                    break

                # Skip already processed batches
                if i * opt.batchSize < processed_count:
                    continue

                if i * opt.batchSize >= opt.how_many:
                    break

                # Process batch
                counter = processed_count + 1
                try:
                    counter = process_batch(model, data_i, output_dir, counter)
                    processed_count = counter - 1
                except Exception as e:
                    print(f"Error processing batch {i}: {str(e)}")
                    continue

            last_dataloader_size = current_dataloader_size

        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            time.sleep(1)
            continue

if __name__ == "__main__":
    main()