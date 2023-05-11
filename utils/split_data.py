import pandas as pd

dataset_loc = '../final_data.csv'

dataset = pd.read_csv(dataset_loc)

disease_dataset = dataset[['image', 'MEL', 'NV', 'BCC', 'AK', 'SL', 'SK', 'LK', 'DF', 'VASC', 'SCC', 'LN', 'CAM', 'AMP', 'UNK']]
metadata_dataset = dataset[['image', 'sex', 'age', 'anatom_site', 'benign_malignant']]

print(len(disease_dataset))
print(len(metadata_dataset))

disease_dataset.to_csv('diseases.csv')
metadata_dataset.to_csv('metadata.csv')