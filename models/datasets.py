import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import pandas as pd

import os

import random

from pathlib import Path
from typing import List

DISEASES = ['MEL', 'NV', 'BCC', 'AK', 'SL', 'SK', 'LK', 'DF', 'VASC', 'SCC', 'LN', 'CAM', 'AMP', 'UNK']
METADATA = ['sex', 'age', 'anatom_site']

def stratified_split_indexes(labels: pd.DataFrame, splits: List[float]):
    labels = labels.reset_index(drop=True)
    labels = labels.sample(frac=1)
    splits = torch.tensor(splits).cumsum(dim=0)
    nclasses = labels.nunique()

    classes = {}
    class_indexes = {}
    for i in range(nclasses):
        classes[i] = list(labels.index[labels==i])
        class_indexes[i] = [0]
        for j in range(len(splits)):
            class_indexes[i].append(int((splits[j]*len(classes[i])).item()))

    split_indexes = [[] for i in range(len(splits))]
    for key in class_indexes.keys():
        for i in range(len(split_indexes)):
            start = class_indexes[key][i]
            end = class_indexes[key][i+1]
            items = classes[key][start:end]
            split_indexes[i].extend(items)

    return tuple(split_indexes)
    
random_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224,224)),
    transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.7, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

resize_tranform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(512),
    transforms.CenterCrop((512,512)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
    
# Classification dataset and metdata dataset for classification and dgan systems
class Classification(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose = random_transform) -> None:
        super().__init__()
        self.transform = transform

        data_path = root / 'all_encoded.csv'
        attn_path = root / 'all_attention.csv'

        labels = pd.read_csv(data_path)
        attn = pd.read_csv(attn_path)

        # We must remove all rows where a disease label does not exist
        # We use the attention mask for this function
        image_names = labels['image']
        disease_labels = labels['disease']
        disease_attn = attn['0']

        samples = disease_attn == 1

        image_names = image_names[samples]

        for i in range(len(image_names)):
            image_names.iloc[i] = Path(root / 'images' / image_names.iloc[i])
        
        self.image_paths = image_names
        self.disease_labels = disease_labels[samples]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths.iloc[index])
        image = self.transform(image)

        disease = self.disease_labels.iloc[index]

        return image, disease 

class Metadata(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose = random_transform) -> None:
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
    # disease_to_str = {
    #     0:'melanoma',
    #     1:'nevus',
    #     2:'basal cell carcinoma',
    #     3:'actinic keratosis',
    #     4:'seborrheic keratosis',
    #     5:'dermatofibroma',
    #     6:'vascular lesion',
    #     7:'squamous cell carcinoma',
    #     8:'lentigo NOS',
    #     9:'unknown'
    # }
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

    def __init__(self, root: Path, transform: transforms.Compose = resize_tranform) -> None:
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

                prompt = f'An image of {bm} {disease} on the {site} of a {age} year old {sex}'
                data['text'].append(prompt)

        self.data = data

        frame = pd.DataFrame(new_labels)
        frame.to_csv(root / 'fake_labels.csv')

    def __len__(self):
        return len(self.data['text'])
    
    def __getitem__(self, index):
        return self.data['text'][index], self.data['image_dir'][index]

        


        
        

        

        












        



    


