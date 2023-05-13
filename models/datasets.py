import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import pandas as pd

from pathlib import Path
from typing import List

DISEASES = ['MEL', 'NV', 'BCC', 'AK', 'SL', 'SK', 'LK', 'DF', 'VASC', 'SCC', 'LN', 'CAM', 'AMP', 'UNK']
METADATA = ['sex', 'age', 'anatom_site']

def stratified_split_indexes(labels: pd.DataFrame, splits: List[float]):
    labels = labels.reset_index(drop=True)
    labels = labels.sample(frac=1)
    print(labels)
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
    disease_to_str = {
        0:'melanoma',
        1:'nevus',
        2:'basal cell carcinoma',
        3:'actinic keratosis',
        4:'solar lentigo',
        5:'seborrheic keratosis',
        6:'lichenoid keratosis',
        7:'dermatofibroma',
        8:'vascular lesion',
        9:'squamous cell carcinoma',
        10:'lentigo NOS',
        11:'cafe-au-lait macule',
        12:'atypical melanocytic proliferation',
        13:'unknown'
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










        



    


