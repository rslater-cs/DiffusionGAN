import torch
from torch.utils.data import Subset, Dataset
import pandas as pd
from typing import List

def stratified_split_indexes(labels: pd.DataFrame, splits: List[float]):
    labels = labels.reset_index(drop=True)
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
        
    
    

