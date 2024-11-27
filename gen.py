import os
import time
import numpy as np
import cv2
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from models.networks.sync_batchnorm import DataParallelWithCallback
import fcntl

def setup_distributed(gpu_id, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=gpu_id
    )
    
    torch.cuda.set_device(gpu_id)

def get_next_batch(rank, batch_size, input_dir, output_dir, lock_file):
    """Get next batch of unprocessed files for this GPU"""
    files = []
    next_file = 1
    
    with open(lock_file, 'a+') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            processed = set(int(x.strip()) for x in f.readlines() if x.strip())
            
            while len(files) < batch_size:
                # Skip processed files
                while next_file in processed:
                    next_file += 1
                
                # Check if file exists
                input_path = os.path.join(input_dir, f"{next_file}.bmp")
                if not os.path.exists(input_path):
                    break
                
                files.append(next_file)
                f.write(f"{next_file}\n")
                f.flush()
                next_file += 1
                
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    return files if files else None

def load_masks_batch(file_nums, input_dir, output_dir):
    """Load a batch of masks."""
    masks = []
    valid_files = []
    
    for file_num in file_nums:
        input_path = os.path.join(input_dir, f"{file_num}.bmp")
        mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append(mask)
            valid_files.append(file_num)
    
    if not masks:
        return None, []
        
    masks_tensor = torch.from_numpy(np.stack(masks)).unsqueeze(1).float()
    
    batch_size = len(masks)
    H, W = masks[0].shape
    
    return {
        'label': masks_tensor,
        'instance': torch.zeros(batch_size),
        'image': torch.zeros(batch_size, 3, H, W),
        'path': [os.path.join(input_dir, f"{num}.bmp") for num in valid_files]
    }, valid_files

def save_batch_images(image_arrays, file_nums, output_dir):
    """Save a batch of images."""
    for img, file_num in zip(image_arrays, file_nums):
        output_path = os.path.join(output_dir, f"{file_num:09}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

def process_batch(model, file_nums, input_dir, output_dir, gpu_id):
    """Process a batch of images."""
    try:
        # Load batch
        data_i, valid_files = load_masks_batch(file_nums, input_dir, output_dir)
        if data_i is None or not valid_files:
            return True
            
        # Move to specific GPU
        data_i = {k: v.cuda(gpu_id) if isinstance(v, torch.Tensor) else v 
                 for k, v in data_i.items()}
            
        # Generate
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = model.module(data_i, mode='inference') if hasattr(model, 'module') \
                      else model(data_i, mode='inference')
            
        # Process and save
        image_arrays = []
        for i in range(len(valid_files)):
            img = ((generated[i].cpu().numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            image_arrays.append(img)
            
        save_batch_images(image_arrays, valid_files, output_dir)
        print(f"GPU {gpu_id}: Processed files {valid_files}")
        
        # Clear memory
        del generated, data_i, image_arrays
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"Error in batch processing on GPU {gpu_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_on_gpu(rank, world_size, opt):
    """Process images on a single GPU."""
    setup_distributed(rank, world_size)
    
    # Initialize model
    model = Pix2PixModel(opt)
    model.eval()
    model.cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create lock file
    lock_file = os.path.join(opt.results_dir, 'processed_files.txt')
    
    try:
        while True:
            # Get next batch of files
            batch_files = get_next_batch(rank, opt.batchSize, 
                                       opt.label_dir, opt.results_dir, lock_file)
            
            if batch_files is None:
                time.sleep(1)
                continue
                
            # Process batch
            success = process_batch(model, batch_files, opt.label_dir, opt.results_dir, rank)
            
            if not success:
                print(f"GPU {rank}: Failed to process batch {batch_files}, retrying...")
                time.sleep(1)
                
    except KeyboardInterrupt:
        print(f"\nStopping GPU {rank}...")
    
    finally:
        # Cleanup
        dist.destroy_process_group()
        del model
        torch.cuda.empty_cache()

def main():
    opt = TestOptions().parse()
    os.makedirs(opt.results_dir, exist_ok=True)
    
    # Initialize lock file
    lock_file = os.path.join(opt.results_dir, 'processed_files.txt')
    open(lock_file, 'a').close()  # Create if doesn't exist
    
    world_size = len(opt.gpu_ids)
    
    if world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(process_on_gpu, args=(world_size, opt), nprocs=world_size)
    else:
        process_on_gpu(0, 1, opt)

if __name__ == "__main__":
    main()
