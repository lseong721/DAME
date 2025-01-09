import os    
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
from dataclasses import dataclass
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
import torch
from diffusers import UNet2DModel
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler, PNDMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
import math
from PIL import Image
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
from accelerate import notebook_launcher
import numpy as np
import random
from glob import glob
import argparse
from data_loader import CustomImageDataset
from models import ViT, MAE
import cv2
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure
    
def train_loop(model, noise_scheduler, optimizer, train_dataloader, test_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, args.log_name, "logs")
    )
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
        
    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler)
    
    global_step = 0

    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(accelerator.device)
    # loss_fn_vgg = lpips.LPIPS(net='vgg').to(accelerator.device)
    # ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(accelerator.device)
    
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            # Sample noise to add to the images
            batch_size = clean_images.shape[0]

            # Sample a random timestep for each image
            noise_scheduler.set_timesteps(noise_scheduler.num_train_timesteps)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=clean_images.device).long()
            
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                pred_images, mask = model(clean_images, noisy_images, timesteps, rand_indices=None)
                loss_mse = F.mse_loss(pred_images, clean_images)
                loss_lpips = lpips(pred_images.clip(-1, 1), clean_images)
                loss = loss_mse + loss_lpips * 2# + loss_ssim
                
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss_mse.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            global_step += 1

        progress_bar.close()

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                batch = next(iter(test_dataloader))
                clean_images = batch['images']
                batch_size = clean_images.shape[0]

                # loss, noisy_images = model(clean_images)

                noisy_images = torch.randn_like(clean_images)

                masked_patches = model.to_patch(clean_images)
                batch, num_patches, *_ = masked_patches.shape

                rand_indices = torch.rand(batch, num_patches, device=accelerator.device).argsort(dim = -1)

                noise_scheduler.set_timesteps(50)
                for i, t in enumerate(tqdm(noise_scheduler.timesteps.to(accelerator.device))):

                    with torch.no_grad():
                        # if i == 0:
                        #     noise_pred, mask = model(clean_images, noisy_images, t.repeat(batch_size), rand_indices=rand_indices)
                        # else:
                        #     pass
                        noise_pred, mask = model(clean_images, noisy_images, t.repeat(batch_size), rand_indices=rand_indices)
                        noisy_images = noise_scheduler.add_noise(noise_pred, noise, t.repeat(batch_size))

                        num_masked = int(model.masking_ratio * num_patches)
                        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
                        batch_range = torch.arange(batch, device=accelerator.device)[:, None]
                        
                        noisy_patches = model.to_patch(noisy_images)
                        noisy_patches[batch_range, unmasked_indices] = masked_patches[batch_range, unmasked_indices]
                        noisy_images = model.encoder.to_image(noisy_patches)
                        
                        # sample = (noisy_images.clip(-1, 1).detach().cpu().numpy() + 1) / 2
                        # cv2.imwrite('outputs/{:3d}.png'.format(i), sample[0].transpose(1, 2, 0) * 255)
                    
                sample = (noisy_images.clip(-1, 1).detach().cpu().numpy() + 1) / 2
                images_processed = (sample * 255).round().astype("uint8")
                accelerator.trackers[0].writer.add_images("test_samples", images_processed[:, :, :, :], epoch+1)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                torch.save(model.state_dict(), os.path.join(args.output_dir, args.log_name, 'model.pth'))
                # pipeline.save_pretrained(config.output_dir) 


@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 500
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    seed = 0
    num_steps = 1000
    patch_size = 16
    mask_ratio = 0.75
    
def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=6)
parser.add_argument('--model_dir', type=str, default='models')
parser.add_argument("--log_name", type=str, default="diffmae")
parser.add_argument("--data_name", type=str, default="test")
parser.add_argument("--data_dir", type=str, default="/hdd5_raidhdd0/Dataset/UVNormalSet")
parser.add_argument("--output_dir", type=str, default="checkpoints")

args = parser.parse_args()
config = TrainingConfig()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)    

if __name__ == '__main__':            
    setup_seed(config.seed)

    os.makedirs(args.model_dir, exist_ok=True)


    config.dataset_name = "huggan/smithsonian_butterflies_subset"
    # dataset = load_dataset(config.dataset_name, split="train[:10%]")
    train_dataset = load_dataset(config.dataset_name, split="train[:80%]")
    test_dataset = load_dataset(config.dataset_name, split="train[80%:]")

    preprocess = transforms.Compose([transforms.Resize((config.image_size, config.image_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5]),])

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    train_dataset.set_transform(transform)
    test_dataset.set_transform(transform)

    noise_scheduler = PNDMScheduler(num_train_timesteps=1000)

    vit = ViT(
        image_size = config.image_size,
        patch_size = config.patch_size,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048
    )

    mae = MAE(
        encoder = vit,
        masking_ratio = 0.25,   # the paper recommended 75% masked patches
        decoder_dim = 512,      # paper showed good results with just 512
        decoder_depth = 6       # anywhere from 1 to 8
    )
    # mae.load_state_dict(torch.load('checkpoints/model.pth', weights_only=True))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.train_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(mae.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=config.lr_warmup_steps,
                                                   num_training_steps=(len(train_dataloader) * config.num_epochs),)

    train_loop(mae, noise_scheduler, optimizer, train_dataloader, test_dataloader, lr_scheduler)