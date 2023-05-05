from utils.datasets import Metadata, Classification
from pathlib import Path

from torch.utils.data import DataLoader

base_path = Path('Z:\ISIC_Combined')
meta = Metadata(root=base_path)
clas = Classification(root=base_path)

metaloader = iter(DataLoader(meta, batch_size=16, shuffle=False))
clasloader = iter(DataLoader(clas, batch_size=16, shuffle=False))

for image, disease in clasloader:
    print(disease)

for image, disease, sex, age, site, bm, attn in metaloader:
    print(attn)
