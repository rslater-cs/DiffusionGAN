import torch
from torch.utils.data import Dataset
from torch import nn, Tensor
from torchvision import transforms
import cv2

from PIL import Image

import pandas as pd
import numpy as np

from pathlib import Path

DISEASES = ['MEL', 'NV', 'BCC', 'AK', 'SL', 'SK', 'LK', 'DF', 'VASC', 'SCC', 'LN', 'CAM', 'AMP', 'UNK']
METADATA = ['sex', 'age', 'anatom_site']

class Numpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor):
        return x.numpy()
    
default_transform = transforms.Compose([
    transforms.ToTensor(),
    Numpy()
])

def zero_disease(labels: pd.DataFrame, label_length: int):
    for disease in DISEASES:
        labels[disease] = np.zeros((label_length,))
    return labels

###########################################################################################################################
# The ISIC datasets are the raw values, this was only used for dataset merging. Not to be used for training
class ISIC(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def expand_labels(self):
        labels = dict.fromkeys(self.full_labelset, value=np.nan)
        return pd.DataFrame(labels)
    
    def __getitem__(self, index):
        image_name = str(self.image_names[index])
        image = cv2.imread(image_name)
        return image, self.labels.iloc[index], self.image_names[index]
    
    
class ISIC_2020(Dataset):
    def __init__(self, root: Path, transform: nn.Module = default_transform) -> None:
        super().__init__()
        self.disease_encoding = {'melanoma':'MEL', 'nevus':'NV', 'unknown':'UNK', 
                                 'seborrheic keratosis':'SK', 'lentigo NOS':'LN', 
                                 'lichenoid keratosis':'SK', 'solar lentigo':'SL', 
                                 'cafe-au-lait macule':'CAM', 
                                 'atypical melanocytic proliferation':'AMP'}

        self.transform = transform

        ground_truth_path = root / 'ISIC_2020_Training_GroundTruth.csv'
        images_path = root / 'ISIC_2020_Training_Input'
        
        self.labels = pd.read_csv(ground_truth_path)
        self.image_names = self.labels['image_name']
        self.labels = self.labels.drop('image_name', axis=1)
        self.labels = self.labels.drop('patient_id', axis=1)

        for i in range(len(self.image_names)):
            self.image_names.iloc[i] = images_path / (self.image_names.iloc[i]+'.jpg')

        self.labels = self.expand_labels()

    def expand_labels(self):
        label_length = len(self.labels)
        full_labelset = ['image'] + DISEASES + METADATA + ['benign_malignant']

        full_labels = dict.fromkeys(full_labelset, np.full((label_length,), np.nan))
        
        full_labels['image'] = self.image_names
        full_labels['sex'] = self.labels['sex']
        full_labels['age'] = self.labels['age_approx']
        full_labels['anatom_site'] = self.labels['anatom_site_general_challenge']
        full_labels['benign_malignant'] = self.labels['target']
        
        full_labels = zero_disease(full_labels, label_length)
        for i in range(label_length):
            full_labels[self.disease_encoding[self.labels['diagnosis'].iloc[i]]][i] = 1.0

        full_labels = pd.DataFrame(full_labels)

        return full_labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_name = str(self.image_names[index])
        image = cv2.imread(image_name)
        return image, self.labels.iloc[index], self.image_names[index]


class ISIC_2019(Dataset):

    def __init__(self, root: Path, transform: nn.Module = default_transform) -> None:
        super().__init__()

        self.transform = transform

        train_ground_truth_path = root / 'ISIC_2019_Training_GroundTruth.csv'
        train_metadata_path = root / 'ISIC_2019_Training_Metadata.csv'
        train_images_path = root / 'ISIC_2019_Training_Input'
        
        ground_truth = pd.read_csv(train_ground_truth_path)

        metadata = pd.read_csv(train_metadata_path)
        metadata = metadata.drop('lesion_id', axis=1)

        self.labels = pd.concat([ground_truth, metadata], axis='columns', join='outer')

        self.image_names = self.labels['image'].iloc[:, 0]
        
        for i in range(len(self.image_names)):
            self.image_names[i] = train_images_path / (self.image_names.iloc[i]+'.jpg')

        self.labels = self.labels.drop('image', axis=1)
        self.labels = self.expand_labels()

    def expand_labels(self):
        label_length = len(self.labels)
        full_labelset = ['image'] + DISEASES + METADATA + ['benign_malignant']

        full_labels = dict.fromkeys(full_labelset, np.full((label_length,), np.nan))
        
        full_labels['image'] = self.image_names
        full_labels['sex'] = self.labels['sex']
        full_labels['age'] = self.labels['age_approx']
        full_labels['anatom_site'] = self.labels['anatom_site_general']
        
        full_labels = zero_disease(full_labels, label_length)

        full_labels['MEL'] = self.labels['MEL']
        full_labels['NV'] = self.labels['NV']
        full_labels['BCC'] = self.labels['BCC']
        full_labels['AK'] = self.labels['AK']
        full_labels['DF'] = self.labels['DF']
        full_labels['VASC'] = self.labels['VASC']
        full_labels['SCC'] = self.labels['SCC']
        full_labels['UNK'] = self.labels['UNK']

        full_labels = pd.DataFrame(full_labels)

        return full_labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_name = str(self.image_names[index])
        image = cv2.imread(image_name)
        return image, self.labels.iloc[index], self.image_names[index]
    
    
class ISIC_2018(Dataset):
    def __init__(self, root: Path, split: str = 'train', transform: nn.Module = default_transform) -> None:
        super().__init__()
        self.splits = ['train', 'valid', 'test']

        self.transform = transform

        if split == 'train':
            ground_truth_path = root / 'ISIC2018_Task3_Training_GroundTruth.csv'
            images_path = root / 'ISIC2018_Task3_Training_Input'
        elif split == 'valid':
            ground_truth_path = root / 'ISIC2018_Task3_Validation_GroundTruth.csv'
            images_path = root / 'ISIC2018_Task3_Validation_Input'
        elif split == 'test':
            ground_truth_path = root / 'ISIC2018_Task3_Test_GroundTruth.csv'
            images_path = root / 'ISIC2018_Task3_Test_Input'
        else:
            raise Exception("Split must be either train, valid or test")
        
        self.labels = pd.read_csv(ground_truth_path)
        self.image_names = self.labels['image']
        self.labels = self.labels.drop('image', axis=1)

        for i in range(len(self.image_names)):
            self.image_names.iloc[i] = images_path / (self.image_names.iloc[i]+'.jpg')

        self.labels = self.expand_labels()

    def expand_labels(self):
        label_length = len(self.labels)
        full_labelset = ['image'] + DISEASES + METADATA + ['benign_malignant']

        full_labels = dict.fromkeys(full_labelset, np.full((label_length,), np.nan))
        
        full_labels['image'] = self.image_names
        
        full_labels = zero_disease(full_labels, label_length)

        full_labels['MEL'] = self.labels['MEL']
        full_labels['NV'] = self.labels['NV']
        full_labels['BCC'] = self.labels['BCC']
        full_labels['AK'] = self.labels['AKIEC']
        full_labels['DF'] = self.labels['DF']
        full_labels['VASC'] = self.labels['VASC']

        full_labels = pd.DataFrame(full_labels)

        return full_labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_name = str(self.image_names[index])
        image = cv2.imread(image_name)
        return image, self.labels.iloc[index], self.image_names[index]
    

class ISIC_2017(Dataset):
    def __init__(self, root: Path, split: str = 'train', transform: nn.Module = default_transform) -> None:
        super().__init__()
        self.splits = ['train', 'valid', 'test']

        self.transform = transform

        if split == 'train':
            ground_truth_path = root / 'ISIC-2017_Training_Part3_GroundTruth.csv'
            images_path = root / 'ISIC-2017_Training_Data'
        elif split == 'valid':
            ground_truth_path = root / 'ISIC-2017_Validation_Part3_GroundTruth.csv'
            images_path = root / 'ISIC-2017_Validation_Data'
        elif split == 'test':
            ground_truth_path = root / 'ISIC-2017_Test_Part3_GroundTruth.csv'
            images_path = root / 'ISIC-2017_Test_Data'
        else:
            raise Exception("Split must be either train, valid or test")
        
        self.labels = pd.read_csv(ground_truth_path)
        self.image_names = self.labels['image_id']
        self.labels = self.labels.drop('image_id', axis=1)

        for i in range(len(self.image_names)):
            self.image_names.iloc[i] = images_path / (self.image_names.iloc[i]+'.jpg')

        self.labels = self.expand_labels()

    def expand_labels(self):
        label_length = len(self.labels)
        full_labelset = ['image'] + DISEASES + METADATA + ['benign_malignant']

        full_labels = dict.fromkeys(full_labelset, np.full((label_length,), np.nan))
        
        full_labels['image'] = self.image_names
        
        full_labels = zero_disease(full_labels, label_length)

        full_labels['MEL'] = self.labels['melanoma']
        full_labels['SK'] = self.labels['seborrheic_keratosis']

        full_labels = pd.DataFrame(full_labels)

        return full_labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_name = str(self.image_names[index])
        image = cv2.imread(image_name)
        return image, self.labels.iloc[index], self.image_names[index]
    

class ISIC_2016(Dataset):
    def __init__(self, root: Path, split: str = 'train', transform: nn.Module = default_transform) -> None:
        super().__init__()
        self.splits = ['train', 'test']

        self.transform = transform

        if split == 'train':
            ground_truth_path = root / 'ISBI2016_ISIC_Part3_Training_GroundTruth.csv'
            images_path = root / 'ISBI2016_ISIC_Part3_Training_Data'
        elif split == 'test':
            ground_truth_path = root / 'ISBI2016_ISIC_Part3_Test_GroundTruth.csv'
            images_path = root / 'ISBI2016_ISIC_Part3_Test_Data'
        else:
            raise Exception("Split must be either train or test")
        
        self.labels = pd.read_csv(ground_truth_path, header=None)
        self.labels = self.labels.rename(columns={0:'image', 1:'mb'})
        self.image_names = self.labels.iloc[:, 0]
        self.labels = self.labels.drop('image', axis=1)

        for i in range(len(self.image_names)):
            self.image_names.iloc[i] = images_path / (self.image_names.iloc[i]+'.jpg')

        self.labels = self.expand_labels()

    def expand_labels(self):
        label_length = len(self.labels)
        full_labelset = ['image'] + DISEASES + METADATA + ['benign_malignant']

        full_labels = dict.fromkeys(full_labelset, np.full((label_length,), np.nan))
        
        full_labels['image'] = self.image_names
        full_labels['benign_malignant'] = map(int, self.labels['mb'] == 'malignant')
        
        full_labels = zero_disease(full_labels, label_length)

        full_labels = pd.DataFrame(full_labels)

        return full_labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_name = str(self.image_names[index])
        image = cv2.imread(image_name)
        return image, self.labels.iloc[index], self.image_names[index]
    
########################################################################################
    
resize_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224,224)),
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
        image = Image.open(self.image_paths[index])
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



    


