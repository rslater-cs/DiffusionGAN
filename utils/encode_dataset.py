import pandas as pd
from pathlib import Path
import numpy as np
import math

anatom_sites = dict.fromkeys(['head/neck', 'upper extremity', 'lower extremity', 'torso', 'palms/soles',
                    'oral/genital', 'anterior torso', 'posterior torso', 'lateral torso'])

for i, key in enumerate(anatom_sites.keys()):
    anatom_sites[key] = i

base_path = Path('Z:\ISIC_Combined')

metadata_path = base_path / 'metadata.csv'
disease_path = base_path / 'diseases.csv'

metadata = pd.read_csv(metadata_path)

encoded_metadata = dict.fromkeys(metadata.columns)
for key in encoded_metadata.keys():
    encoded_metadata[key] = np.empty((len(metadata,)))

encoded_metadata['image'] = metadata['image']
for i in range(len(metadata)):
    encoded_metadata['sex'][i] = 0 if metadata['sex'].iloc[i] == 'male' else 1
    encoded_metadata['age'][i] = int(metadata['age'].iloc[i]//5) if not math.isnan(metadata['age'].iloc[i]) else metadata['age'].iloc[i]
    encoded_metadata['anatom_site'][i] = anatom_sites.get(metadata['anatom_site'].iloc[i], np.nan)
    encoded_metadata['benign_malignant'][i] = metadata['benign_malignant'].iloc[i]

encoded_metadata = pd.DataFrame(encoded_metadata)
encoded_metadata.to_csv('./e_metadata.csv', index=False)