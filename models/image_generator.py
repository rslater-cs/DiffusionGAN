%pip install diffusers
%pip install transformers
%pip install diffusers==0.12.1
%pip install open_clip_torch
%pip install datasets==2.10.1

import os
import csv
import numpy

from PIL import Image
from tqdm.auto import tqdm

import torch
from torch import autocast
from torch.nn.functional import mse_loss

from torchvision import transforms

import datasets
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler, get_cosine_schedule_with_warmup

import open_clip

dataset_dir = os.getcwd() + "/dataset"
transformer_dir = cache_dir=os.getcwd() + "/transformer"

manual_datset = False

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def decode_image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    
    return pil_images[0]

class DiffusionModel:
    def __init__(self, device):
        self.device = device
        
        self.width = 512                         # default width of Stable Diffusion
        self.height = 512                        # default height of Stable Diffusion

        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", cache_dir=transformer_dir, subfolder="vae", sample_size=1).to(self.device)

        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", cache_dir=transformer_dir, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1", cache_dir=transformer_dir, subfolder="text_encoder")

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", cache_dir=transformer_dir, subfolder="unet").to(self.device)
            
        # 4. The scheduler for managing the denoising amount
        self.scheduler = LMSDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", cache_dir=transformer_dir, subfolder="scheduler")
        self.scheduler.set_timesteps(50)
        
        self.generator = torch.manual_seed(1)
        
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        
        # default_lr = 5e-6
        
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=5e-5,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
        
    def prompts_to_embeddings(self, prompts):
        # Embedding
        text_input = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids)
            
        return text_embeddings
        
    def prompts_to_query_embeddings(self, prompts):
        text_input = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids)
        
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)
        
        return torch.cat([uncond_embeddings[0], text_embeddings[0]])
    
    def generate_latent_noise(self, num):
        shape = (num, self.unet.in_channels, self.height // 8, self.width // 8)
        
        latents = torch.randn(shape, generator=self.generator)
        
        latents = latents * self.scheduler.init_noise_sigma
        
        return latents
    
    def generate_step(self, step_index, latents, text_embeddings, guidance_scale):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = self.scheduler.scale_model_input(latent_model_input, step_index)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = self.unet(latent_model_input, step_index, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        return self.scheduler.step(noise_pred, step_index, latents).prev_sample
            
    def generate_raw(self, latents, text_embeddings):
        guidance_scale = 7.5                # Scale for classifier-free guidance

        latents = latents.to(self.device)
        
        text_embeddings = text_embeddings.to(self.device)
        
        for step_index in tqdm(self.scheduler.timesteps):
            latents = self.generate_step(step_index, latents, text_embeddings, guidance_scale)
            
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample
            
        return image
    
    def generate(self, prompt):
        latents = self.generate_latent_noise(1)
    
        text_embeddings = self.prompts_to_query_embeddings([prompt])
    
        return decode_image(self.generate_raw(latents, text_embeddings))
    
    def show_image(self, latents):
        with torch.no_grad():
            image = self.vae.decode(latents).sample
            
        image_grid([decode_image(image)], 1, 1).show()
            
    def train_step(self, batch, lr_scheduler):
        images = batch["image"].to(self.device)
        
        with torch.no_grad():
            # Convert image to latents
            image_latents = self.vae.encode(images.float().cuda()).latent_dist.sample()
        
            # Generate image noise
            noise = torch.randn(image_latents.shape).to(self.device)

            # Generate random timesteps
            bsz = image_latents.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()
        
            # Generate noisy versions of images based on timesteps
            noise_latents = self.noise_scheduler.add_noise(image_latents, noise, timesteps)
        
            text_embeddings = self.prompts_to_embeddings(batch["text"])
        
        predicted_latents = self.unet(noise_latents, timesteps, encoder_hidden_states=text_embeddings.last_hidden_state.cuda()).sample
        
        #self.show_image(image_latents)
        #self.show_image(noise)
        #self.show_image(noise_latents)
        #self.show_image(predicted_latents)
            
        loss = mse_loss(predicted_latents, noise, reduction="mean")
        
        loss.backward()
        
        self.optimizer.step()
        lr_scheduler.step()
        self.optimizer.zero_grad()
        
    def train_go_emotions(self, dataset, test_func):
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
        
        epochs = 3
        
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(train_dataloader) * epochs),
        )
        
        for i in range(epochs):
            for batch in tqdm(train_dataloader):
                # Change the image to be (3, width, height) format
                #image = entry["image"].permute(2, 0, 1)
                self.train_step(batch, lr_scheduler)
                
            test_func()

# Manual dataset if datasets module isn't able to download anything
class Dataset:
    def __init__(self, filename):
        self.entries = []
        
        with open(filename, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            
            csv_reader.__next__()
            
            for _, file_name, dx, dx_type, age, sex, localization in csv_reader:
                self.entries.append({"file_name": file_name, "dx": dx, "dx_type": dx_type, "age": age, "sex": sex, "localization": localization})
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        try:
            img = Image.open("download/HAM10000_images_part_1/" + entry["file_name"] + ".jpg")
        except Exception as e:
            img = Image.open("download/HAM10000_images_part_2/" + entry["file_name"] + ".jpg")
            
        convert_tensor = transforms.ToTensor()
        
        entry["image"] = convert_tensor(img)
        
        if self.transform:
            entry = self.transform(entry)
        
        return entry
    
    def set_transform(self, transform):
        self.transform = transform

if False:
    dataset = Dataset("download/HAM10000_metadata.csv")
    
    transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512), # Makes sense for these images in particular
        transforms.RandomHorizontalFlip()
    ])

    def make_prompt(entry):
        dx = entry["dx"]
        dx_type = entry["dx_type"]
        age = entry["age"]
        sex = entry["sex"]
        position = entry["localization"]

        return f"An image of {dx} skin cancer of type {dx_type} located at {position} on a {sex} of age {age}"

    def transform_images(entry):
        new_entry = {}
        new_entry["image"] = transform(entry["image"])
        new_entry["text"] = make_prompt(entry)
        return new_entry

    dataset.set_transform(transform_images)

else:
    # datsets that automatically downloads datasets
    dataset = datasets.load_dataset("marmal88/skin_cancer", cache_dir=dataset_dir).with_format("torch").cast_column("image", datasets.Image(decode=True))["train"]

    transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512), # Makes sense for these images in particular
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    def transform_images(entries):
        new_entries = {}
        new_entries["image"] = [transform(entry.convert("RGB")) for entry in entries["image"]]
        new_entries["text"] = []
        
        for i in range(len(entries["dx"])):
            dx = entries["dx"][i]
            dx_type = entries["dx_type"][i]
            age = entries["age"][i]
            sex = entries["sex"][i]
            position = entries["localization"][i]

            new_entries["text"].append(f"An image of {dx} skin cancer of type {dx_type} located at {position} on a {sex} of age {age}")
            
        return new_entries

    dataset.set_transform(transform_images)

model = DiffusionModel(torch_device)

def test_func():
    images = []
    for i in range(0, 4):
        image = model.generate("actinic_keratoses skin cancer of type histo located on the foot of a male of age 50.0")

        #image = model.generate("skin cancer")

        #image = model.generate("an image of a dog wearing a hat")

        images.append(image)

    image_grid(images, 2, 2).show()

model.train_go_emotions(dataset, test_func)

test_func()