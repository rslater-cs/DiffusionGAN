import pandas as pd
from pathlib import Path

data_path = Path('Z:\\ISIC_Combined')
meta_path = data_path / 'metadata.csv'
disease_path = data_path / 'diseases.csv'
all_path = data_path / 'final_data.csv'

metadata = pd.read_csv(meta_path)

for i, row in metadata.iterrows():
    metadata['image'].iloc[i] = Path(row['image']).name

disease_data = pd.read_csv(disease_path)

for i, row in disease_data.iterrows():
    disease_data['image'].iloc[i] = Path(row['image']).name

all_data = pd.read_csv(all_path)

for i, row in all_data.iterrows():
    all_data['image'].iloc[i] = Path(row['image']).name


metadata.to_csv(meta_path, index=False)
disease_data.to_csv(disease_path, index=False)
all_data.to_csv(all_path, index=False)