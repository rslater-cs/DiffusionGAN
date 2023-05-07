from utils.datasets import Classification, Metadata
from pathlib import Path

from torch.utils.data import DataLoader

base_path = Path('Z:\ISIC_Combined')
meta = Metadata(root=base_path)
clas = Classification(root=base_path)

metaloader = iter(DataLoader(meta, batch_size=4, shuffle=False))
clasloader = iter(DataLoader(clas, batch_size=4, shuffle=False))

# for image, disease in clasloader:
#     print(disease)

for image, disease, sex, age, site, bm, attn in metaloader:
    print(disease)
    print(sex)
    print(age)
    print(site)
    print(bm)
    print(attn)
