from utils.datasets import Metadata
from pathlib import Path

base_path = Path('./')
# clas = Classification(root=base_path)
meta = Metadata(root=base_path)

attn = meta.attn[meta.attn['0'] == 1]
attn = attn[attn['4'] == 1]

for i in range(14):
    disease = meta.labels[(meta.labels['disease'] == i)]
    disease = disease[disease.index.isin(attn.index)]
    b = (disease['benign_malignant'] == 0).all()
    m = (disease['benign_malignant'] == 1).all()
    print(disease)
    print("Benign", b)
    print("Malignant", m)
# bcc = bcc[(meta.attn['0'] == 1).index & bcc.index]
# scc = meta.labels[meta.labels]
# print(bcc)
