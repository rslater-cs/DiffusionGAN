from datasets import ISIC_2020, ISIC_2019, ISIC_2018, ISIC_2017, ISIC_2016
from pathlib import Path

import numpy as np

import cv2

base = Path('D:\\ISIC_Dataset\\NetworkDatasets\\DiseaseClassification')

def dhash(image, hash_size=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size+1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]

    hash_val = sum([2**i for (i,v) in enumerate(diff.flatten()) if v])
    return hash_val


if __name__ == '__main__':
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

    image_hashes = dict()

    progress = 0

    for image, labels, loc in iter(isic_2020):
        if progress % 100 == 0:
            print(progress)
        progress += 1
        
        image_hash = dhash(image)

        duplicates = image_hashes.get(image_hash, [])

        if len(duplicates) == 0:
            image_hashes[image_hash] = [loc]
        else:
            image_hashes[image_hash].append(loc)
            print(image_hashes[image_hash])


