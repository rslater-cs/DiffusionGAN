#%pip install diffusers
#%pip install transformers
#%pip install diffusers==0.12.1
#%pip install open_clip_torch
#%pip install datasets==2.10.1

#cd ./.. # Move to root directory
#ls -a # list directory and hidden items

import os
import csv
import numpy
import pandas as pd
import random

from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
from torch import autocast
from torch.utils.data import Dataset
from torch.nn.functional import mse_loss

from torchvision import transforms

import datasets
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler, get_cosine_schedule_with_warmup

import open_clip

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def decode_images(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    
    return pil_images

def test_func(model, prompt, negative_prompt):
    images = []
    transform = transforms.Resize(100)
    for i in range(0, 4):
        for image in model.generate([prompt] * 8, [negative_prompt] * 8):
            images.append(transform(image))

    image_grid(images, 4, 8).show()

class DiffusionModel:
    def __init__(self, device, path = None):
        self.device = device
        
        self.width = 512                         # default width of Stable Diffusion
        self.height = 512                        # default height of Stable Diffusion

        transformer_dir = os.getcwd() + "/transformer"
        
        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", cache_dir=transformer_dir, subfolder="vae", sample_size=1).to(self.device)

        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", cache_dir=transformer_dir, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1", cache_dir=transformer_dir, subfolder="text_encoder")

        # 3. The UNet model for generating the latents.
        if path:
            self.unet = UNet2DConditionModel.from_pretrained(path).to(self.device)
        else:
            self.unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", cache_dir=transformer_dir, subfolder="unet").to(self.device)
            
        # 4. The scheduler for managing the denoising amount
        self.scheduler = LMSDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", cache_dir=transformer_dir, subfolder="scheduler")
        
        self.generator = torch.manual_seed(1)
        
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        
        # default_lr = 5e-6
        
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=5e-6,
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
        
    def prompts_to_query_embeddings(self, prompts, negative_prompts):
        text_input = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids)
        
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(negative_prompts, padding="max_length", max_length=max_length, return_tensors="pt")
        
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
    
    def generate(self, prompts, negative_prompts, steps):
        self.scheduler.set_timesteps(steps)
        
        latents = self.generate_latent_noise(len(prompts))
    
        text_embeddings = self.prompts_to_query_embeddings(prompts, negative_prompts)
    
        return decode_images(self.generate_raw(latents, text_embeddings))
    
    def show_image(self, latents):
        with torch.no_grad():
            image = self.vae.decode(latents).sample
            
        image_grid([decode_image(image)], 1, 1).show()
            
    def train_step(self, batch, lr_scheduler):
        images = batch["image"].to(self.device)
        
        # Convert image to latents
        with torch.no_grad():
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
        
    def train(self, dataset, epochs, batch_size):
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
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
            
    def save(self, path):
        self.unet.save_pretrained(path)

# Manual dataset if datasets module isn't able to download anything
class HAMDataset:
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

class Metadata(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        super().__init__()

        self.transform = transform
        data_path = root / 'all_encoded.csv'
        attn_path = root / 'all_attention.csv'
        self.labels = pd.read_csv(data_path)

        self.attn = pd.read_csv(attn_path)

        self.attn = self.attn.drop('image', axis=1)

        image_names = self.labels['image']

        for i in range(len(image_names)):
            image_names.iloc[i] = root / 'images' / image_names.iloc[i]

        self.image_paths = image_names
        # Remove all rows where there are no labels at all, this step is now redundant due to 
        # the revised encoding process
        samples = self.attn.sum(axis=1) > 0
        self.image_paths = self.image_paths[samples]

        self.labels = self.labels[samples]

        self.attn = self.attn[samples]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths.iloc[index])
        image = self.transform(image)

        return (image, 
                self.labels['disease'].iloc[index], 
                self.labels['sex'].iloc[index], 
                self.labels['age'].iloc[index], 
                self.labels['anatom_site'].iloc[index],
                self.labels['benign_malignant'].iloc[index],
                torch.tensor(self.attn.iloc[index].values).type(torch.FloatTensor))

class ImageTextPrompt(Metadata):
    disease_to_str = {
        0:'type A',
        1:'type B',
        2:'type C',
        3:'type D',
        4:'type E',
        5:'type F',
        6:'type G',
        7:'type H',
        8:'type I',
        9:'type J'
    }

    sex_to_str = {
        0:'male',
        1:'female'
    }

    age_to_str = lambda s, x: str(int(x*5))

    site_to_str = {
        0:'head/neck',
        1:'upper extremity',
        2:'lower extremity',
        3:'torso',
        4:'palms/soles',
        5:'oral/genital',
        6:'anterior torso',
        7:'posterior torso',
        8:'lateral torso',

    }

    bm_to_str = {
        0:'benign',
        1:'malignant'
    }

    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        super().__init__(root, transform)

    def __getitem__(self, index):
        image = Image.open(self.image_paths.iloc[index])
        image = self.transform(image)
        attn = self.attn.iloc[index]
        labels = self.labels.iloc[index]

        prompt = 'An image of '
        # place bening/malignant into string
        if attn['4'] == 1:
            prompt += f'{self.bm_to_str[labels["benign_malignant"]]} '
            if attn['0'] == 0:
                prompt += 'cancer '
                
        # place disease into string
        if attn['0'] == 1:
            prompt += f'{self.disease_to_str[labels["disease"]]} '
            
        # prepare if we have metadata
        if attn['1'] == 1 or attn['2'] == 1 or attn['3'] == 1:
            prompt += 'on '
            
        # place site into string
        if attn['3'] == 1:
            prompt += f'the {self.site_to_str[labels["anatom_site"]]} of '
            
        # prepare if we have metadata
        if attn['1'] == 1 or attn['2'] == 1 or attn['3'] == 1:
            prompt += 'a '
            
        # place age into string
        if attn['2'] == 1:
            prompt += f'{self.age_to_str(labels["age"])} year old '
        # place sex into string
        
        if attn['1'] == 1:
            prompt += f'{self.sex_to_str[labels["sex"]]}'
            
        data = {'image':image, 'text':prompt}

        return data

class TextPrompt(Dataset):
    disease_to_str = {
        0:'type A',
        1:'type B',
        2:'type C',
        3:'type D',
        4:'type E',
        5:'type F',
        6:'type G',
        7:'type H',
        8:'type I',
        9:'type J'
    }

    sex_to_str = {
        0:'male',
        1:'female'
    }

    age_to_str = lambda s, x: str(int(x*5))

    site_to_str = {
        0:'head/neck',
        1:'upper extremity',
        2:'lower extremity',
        3:'torso',
        4:'palms/soles',
        5:'oral/genital',
        6:'anterior torso',
        7:'posterior torso',
        8:'lateral torso',

    }

    bm_to_str = {
        0:'benign',
        1:'malignant'
    }

    def __init__(self, root: Path, num_images: int) -> None:
        super().__init__()

        data_path = root / 'all_encoded.csv'
        attn_path = root / 'all_attention.csv'
        labels = pd.read_csv(data_path)

        labels = labels['disease']

        attn = pd.read_csv(attn_path)

        attn = attn.drop('image', axis=1)

        if not os.path.exists(str(root) + '/fake_images'):
            os.mkdir(str(root) + '/fake_images')

        data = {'image_dir':[], 'text':[]}

        new_labels = {'image':[],'sex':[],'age':[],'anatom_site':[],'benign_malignant':[],'disease':[]}

        for i in range(num_images):
            data['image_dir'].append(str(root / 'fake_images' / f'ISIC_Fake_{i}.jpg'))

            new_labels['image'].append(f'ISIC_Fake_{i}.jpg')

        distribution = labels.value_counts().sort_index()

        mean = distribution.mean()

        distribution = mean-distribution
        distribution[distribution < 0] = 0
        tot = distribution.sum()

        distribution = distribution / tot
        print(distribution)

        distribution = distribution * num_images
        distribution = distribution.round()

        distribution = distribution.astype(int)

        if distribution.sum() > num_images:
            distribution.iloc[2] += num_images - distribution.sum()

        distribution = list(distribution.values)

        for i, num_samples in enumerate(distribution):
            for j in range(num_samples):
                new_labels['disease'].append(i)

                disease = self.disease_to_str[i]

                age = random.randint(0, 18)
                new_labels['age'].append(age)

                age = self.age_to_str(age)

                sex = random.randint(0, 1)
                new_labels['sex'].append(sex)

                sex = self.sex_to_str[sex]

                site = random.randint(0, 8)
                new_labels['anatom_site'].append(site)

                site = self.site_to_str[site]

                bm = random.randint(0, 1)
                new_labels['benign_malignant'].append(bm)

                bm = self.bm_to_str[bm]

                prompt = f'{bm} {disease} {site} {age} {sex}'
                data['text'].append(prompt)

        self.data = data
        frame = pd.DataFrame(new_labels)

        frame.to_csv(root / 'fake_labels.csv')

    def __len__(self):
        return len(self.data['text'])

    def __getitem__(self, index):
        return self.data['text'][index], self.data['image_dir'][index]

def dataset1():
    resize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(512),
        transforms.CenterCrop((512,512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageTextPrompt(Path("new_dataset/"), resize_transform)
    
    return dataset

def dataset2():
    dataset = HAMDataset("ham10000/HAM10000_metadata.csv")
    
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
    
    return dataset

def dataset3():
    dataset_dir = os.getcwd() + "/marmal"
    
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

            new_entries["text"].append(f"An image of {dx} skin cancer of type {dx_type}. The skin cancer is located at {position} on a {sex} of age {age}")
            
        return new_entries

    dataset.set_transform(transform_images)
    
    return dataset

def train_dataset():
    dataset = dataset1()
    
    model = DiffusionModel(torch_device)
    
    a6000 = 5
    a100 = 13
    
    model.train(dataset, 1, a100)
    
    model.save("save")
    
    test_func(model, "dog", "")
    test_func(model, "actinic_keratoses skin cancer of type histo located on the foot of a male of age 50.0", "deformed body features disfigured blurry dot pattern hairs lines")
    test_func(model, "skin cancer", "deformed body features disfigured blurry dot pattern hairs lines")
    test_func(model, "actinic_keratoses", "deformed body features disfigured blurry dot pattern hairs lines tile bacteria root")

def generate_images(batch_size, image_num):
    model = DiffusionModel(torch_device, "save")
    
    dataset = TextPrompt(Path("new_dataset"), image_num)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    negative_prompt = "deformed body features disfigured blurry line hair dot pattern repeat"
    negative_prompts = [negative_prompt] * batch_size

    transform = transforms.Resize(224)
    
    for prompts, file_paths in iter(dataloader):
        images = model.generate(prompts, negative_prompts, 100)
        
        for i in range(0, batch_size):
            transform(images[i]).save(file_paths[i])

#train_dataset()

#generate_images(24, 2000)