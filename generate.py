"""
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import os
import torch
import cv2
import numpy as np
from collections import OrderedDict
from options.test_options import TestOptions
import data
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from PIL import Image
import time

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)


screen_width, screen_height = 1920, 1080

def resize_with_aspect_ratio(img, screen_width, screen_height):
    h, w, _ = img.shape
    aspect_ratio_img = w / h
    aspect_ratio_screen = screen_width / screen_height

    if aspect_ratio_img > aspect_ratio_screen:
        new_w = screen_width
        new_h = int(screen_width / aspect_ratio_img)
    else:
        new_h = screen_height
        new_w = int(screen_height * aspect_ratio_img)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    centered_img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    y_offset = (screen_height - new_h) // 2
    x_offset = (screen_width - new_w) // 2
    centered_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return centered_img

def blend_images(img1, img2, alpha):
    return cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)

num_steps = 30 # Number of transition frames between two images
delay = 60 # Delay between frames in milliseconds

window_name = 'Generated Image Test'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

image_index = 0
prev_img = None
prev_image_path = None

if not os.path.exists(opt.results_dir):
    os.makedirs(opt.results_dir)

while True:
    for i, data_i in enumerate(dataloader):
        if i * opt.batchSize >= opt.how_many:
            break
        
        with torch.no_grad():
            generated = model(data_i, mode='inference')
        
        current_display_size = (opt.crop_size, int(opt.crop_size * opt.aspect_ratio))
        
        for b in range(generated.shape[0]):
            print(f'Generating image for {data_i["path"][b]}')

            synthesized_image = torch.nn.functional.interpolate(generated[b].unsqueeze(0), size=current_display_size, mode='bilinear').squeeze(0)

            synthesized_image_np = (synthesized_image.permute(1, 2, 0).cpu().numpy() + 1) / 2.0 * 255.0
            synthesized_image_np = synthesized_image_np.astype('uint8')

            output_image_path = os.path.join(opt.results_dir, f'synthesized_image_{image_index}.png')
            Image.fromarray(synthesized_image_np).save(output_image_path)
            
            print(f"Saved image to {output_image_path}")

            img = cv2.imread(output_image_path)
            if img is None:
                print(f"Error: Failed to load file {output_image_path}. Check the path.")
            else:
                resized_img = resize_with_aspect_ratio(img, screen_width, screen_height)

                if prev_img is None:
                    prev_img = resized_img
                    prev_image_path = output_image_path
                    cv2.imshow(window_name, resized_img)
                    cv2.waitKey(1000)
                else:
                    for step in range(num_steps):
                        alpha = step / num_steps
                        blended_image = blend_images(prev_img, resized_img, alpha)
                        cv2.imshow(window_name, blended_image)
                        if cv2.waitKey(delay) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            exit()

                    if prev_image_path:
                        if os.path.exists(prev_image_path):
                            os.remove(prev_image_path)
                            print(f"Deleted previous image file: {prev_image_path}")
                    
                    prev_img = resized_img
                    prev_image_path = output_image_path

            image_index += 1

cv2.destroyAllWindows()