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

class ISIC(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def expand_labels(self):
        labels = dict.fromkeys(self.full_labelset, value=np.nan)
        return pd.DataFrame(labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_name = str(self.image_names[index])
        image = cv2.imread(image_name)
        return image, self.labels.iloc[index], self.image_names[index]
    
class ISIC_2020(ISIC):
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


class ISIC_2019(ISIC):

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
    
class ISIC_2018(ISIC):
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

class ISIC_2017(ISIC):
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

class ISIC_2016(ISIC):
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
