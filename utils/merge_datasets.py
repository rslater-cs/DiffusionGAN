from raw_datasets import ISIC_2020, ISIC_2019, ISIC_2018, ISIC_2017, ISIC_2016, DISEASES, METADATA
from pathlib import Path

import numpy as np

import pandas as pd

import cv2

import shutil
import os

# Specific path of labels and images
base = Path('E:\\Programming\\Datasets\\DiseaseClassification')
save_dir = Path(f'E:\\Programming\\Datasets\\All_ISIC\\')

# All labels that we are merging to
all_labels = ['image'] + DISEASES + METADATA + ['benign_malignant']

# Copies a row of diseases into our new dataset
def copy_diseases(dataset, key, row):
    for disease in DISEASES:
        dataset[disease][key] = row[disease]
    return dataset

# If there are no disease columns with a 1 in them we know that a disease is not present on this row
def has_disease(dataset, key):
    for disease in DISEASES:
        if dataset[disease][key] == 1.0:
            return True
    return False

# There are cases of duplicate images with different labels, we prioitise the newer data if it exists
def manage_conflict(dataset, row, key):
    if not has_disease(dataset, key):
        dataset = copy_diseases(dataset, key, row)
    
    if dataset['sex'][key] == np.nan:
        dataset['sex'][key] = row['sex']

    if dataset['age'][key] == np.nan:
        dataset['age'][key] = row['age']

    if dataset['anatom_site'][key] == np.nan:
        dataset['anatom_site'][key] = row['anatom_site']

    if dataset['benign_malignant'][key] == np.nan:
        dataset['benign_malignant'][key] = row['benign_malignant']
    
    return dataset

def add_row(dataset, labels):
    for column_name in all_labels:
        item = labels[column_name]
        dataset[column_name].append(item)

    return dataset

# Hash the images into a scalar value for efficient comparison
def dhash(image, hash_size=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size+1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]

    hash_val = sum([2**i for (i,v) in enumerate(diff.flatten()) if v])
    return hash_val

def merge_dataset(dataset, new_dataset, image_mapping, image_key):
    for i in range(len(dataset)):
        image, labels, loc = dataset[i]

        if image_key % 100 == 0:
            print(image_key)
        
        # Hash image into a scalar value
        image_hash = dhash(image)

        # If hash exists in our dictionary then the image is a duplicate
        duplicates = image_hashes.get(image_hash, None)

        if duplicates == None:
            # If image is unique then we want to remeber the index key of the image
            image_hashes[image_hash] = image_key

            # We want to change the path to the image to be our new file name
            image_path = labels['image']
            labels = labels.drop('image', axis=0)
            labels['image'] = f'ISIC_Merged_{image_key}.jpg'

            # add the label data to our new dataset
            new_dataset = add_row(new_dataset, labels)

            # copy the image into the new image folder
            shutil.copy(image_path, save_dir/labels['image'])

            # Keeping track of duplicate images in case we need to refer to this data later on
            image_mapping['image_key'].append(image_key)
            image_mapping['original_path'].append(image_path)

            image_key += 1
        else:
            # find the index of the first unique image
            key = image_hashes[image_hash]
            print(key)

            # add the original images index and the duplicate images path to a file that keeps track of mergers 
            image_mapping['image_key'].append(key)
            image_mapping['original_path'].append(labels['image'])

            # try to add the label data of the duplicate image to the dataset, if the dataset has more recent data nothing will be changed
            new_dataset = manage_conflict(new_dataset, labels, key)

    return new_dataset, image_mapping, image_key

if __name__ == '__main__':
    # Load all datasets into a list, order is important as the newer data takes precedent 
    isic_2020 = ISIC_2020(root=base/'2020')

    isic_2019 = ISIC_2019(root=base/'2019')

    isic_2018_train = ISIC_2018(root=base/'2018', split='train')
    isic_2018_valid = ISIC_2018(root=base/'2018', split='valid')
    isic_2018_test = ISIC_2018(root=base/'2018', split='test')

    isic_2017_train = ISIC_2017(root=base/'2017', split='train')
    isic_2017_valid = ISIC_2017(root=base/'2017', split='valid')
    isic_2017_test = ISIC_2017(root=base/'2017', split='test')

    isic_2016_train = ISIC_2016(root=base/'2016', split='train')
    isic_2016_test = ISIC_2016(root=base/'2016', split='test')

    datasets = [
        isic_2020,
        isic_2019,
        isic_2018_train,
        isic_2018_valid,
        isic_2018_test,
        isic_2017_train,
        isic_2017_valid,
        isic_2017_test,
        isic_2016_train,
        isic_2016_test
    ]

    # initialise the dictionaries for labels and merged images

    image_hashes = dict()

    new_dataset = dict.fromkeys(all_labels)
    for column_name in all_labels:
        new_dataset[column_name] = []

    image_mapping = dict.fromkeys(['image_key', 'original_path'])
    for column in image_mapping.keys():
        image_mapping[column] = []

    image_key = 0

    # remove all images in image folder to start from new
    if os.path.exists(f'E:\\Programming\\Datasets\\All_ISIC\\'):
        shutil.rmtree(Path(f'E:\\Programming\\Datasets\\All_ISIC\\'))
    os.mkdir(f'E:\\Programming\\Datasets\\All_ISIC\\')

    # Combine all datasets
    for dataset in datasets:
        print(dataset.__class__)
        new_dataset, image_mapping, image_key = merge_dataset(dataset, new_dataset, image_mapping, image_key)

    # Convert and save to csv file
    final_data = pd.DataFrame(new_dataset)
    final_image_mapping = pd.DataFrame(image_mapping)

    final_data.to_csv('final_data.csv')
    final_image_mapping.to_csv('merged_labels.csv')




