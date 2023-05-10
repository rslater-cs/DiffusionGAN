import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import pandas as pd

from pathlib import Path

DISEASES = ['MEL', 'NV', 'BCC', 'AK', 'SL', 'SK', 'LK', 'DF', 'VASC', 'SCC', 'LN', 'CAM', 'AMP', 'UNK']
METADATA = ['sex', 'age', 'anatom_site']
    
resize_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224,224)),
    transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.7, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
    
# Classification dataset and metdata dataset for classification and dgan systems
class Classification(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose = resize_transform) -> None:
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
    def __init__(self, root: Path, transform: transforms.Compose = resize_transform) -> None:
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



    


