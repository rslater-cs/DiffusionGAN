from datasets import ISIC_2020, ISIC_2019, ISIC_2018, ISIC_2017, ISIC_2016
from pathlib import Path

print("2020")
dataset = ISIC_2020(Path('D:\\ISIC_Dataset\\NetworkDatasets\\DiseaseClassification\\2020'))
print(dataset[0][1])
print("=========================================================")

print("2019")
dataset = ISIC_2019(Path('D:\\ISIC_Dataset\\NetworkDatasets\\DiseaseClassification\\2019'))
print(dataset[0][1])
print("=========================================================")

print("2018")
dataset = ISIC_2018(Path('D:\\ISIC_Dataset\\NetworkDatasets\\DiseaseClassification\\2018'), split='test')
print(dataset[0][1])
print("=========================================================")

print("2017")
dataset = ISIC_2017(Path('D:\\ISIC_Dataset\\NetworkDatasets\\DiseaseClassification\\2017'), split='train')
print(dataset[0][1])
print("=========================================================")

print("2016")
dataset = ISIC_2016(Path('D:\\ISIC_Dataset\\NetworkDatasets\\DiseaseClassification\\2016'), split='train')
print(dataset[0][1])
print("=========================================================")

# dataset = ISIC_2020(Path('D:\\ISIC_Dataset\\NetworkDatasets\\DiseaseClassification\\2020'))
# diagnoses = dataset.labels['diagnosis']

# diseases = dict()

# for row in diagnoses:
#     if row in diseases:
#         diseases[row] += 1
#     else:
#         diseases[row] = 1

# print(diseases)