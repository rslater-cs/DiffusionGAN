import pandas as pd

dataset_loc = './final_data.csv'

dataset = pd.read_csv(dataset_loc)

disease_dataset = dataset[['image', 'MEL', 'NV', 'BCC', 'AK', 'SL', 'SK', 'LK', 'DF', 'VASC', 'SCC', 'LN', 'CAM', 'AMP', 'UNK']]
metadata_dataset = dataset[['image', 'sex', 'age', 'anatom_site', 'benign_malignant']]

print(len(disease_dataset))
print(len(metadata_dataset))

for key in disease_dataset.keys():
    print(len(disease_dataset[key]))
for key in metadata_dataset.keys():
    print(len(metadata_dataset[key]))

disease_dataset.to_csv('diseases.csv', index=False)
metadata_dataset.to_csv('metadata.csv', index=False)