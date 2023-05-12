import pandas as pd
from pathlib import Path
import numpy as np
import math

anatom_sites = dict.fromkeys(['head/neck', 'upper extremity', 'lower extremity', 'torso', 'palms/soles',
                    'oral/genital', 'anterior torso', 'posterior torso', 'lateral torso'])

for i, key in enumerate(anatom_sites.keys()):
    anatom_sites[key] = i

# disease is converted from one hot encoding to index values as is more efficient for cross entropy
def disease_to_int(diseases: pd.Series):
    diseases = diseases.drop('image')
    attn = diseases.sum()

    if attn == 0:
        return 0, 0

    diseases = diseases.values
    # diseases: np.ndarray = diseases.to_numpy(dtype=np.int8)
    index = diseases.nonzero()

    return index[0][0], 1

# simple conversion
def sex_to_int(sex):
    if sex != 'male' and sex != 'female':
        return 0, 0
    
    value = 0 if sex == 'male' else 1

    return value, 1

# divide by 5 to get the node index, using discrete classes will pose issues with new
# data, possibly should discuss this in a poster?
def age_to_int(age):
    if math.isnan(age):
        return 0, 0
    
    value = int(age//5)

    return value, 1

# simple conversion using a dictionary to move from string to int
def anatom_site_to_int(site):
    if site.__class__ != str and math.isnan(site):
        return 0, 0
    
    value = anatom_sites[site]

    return value, 1

# simple conversion
def benign_malignant_to_int(bm):
    if math.isnan(bm):
        return 0, 0
    
    return int(bm), 1

base_path = Path('./')

metadata_path = base_path / 'metadata.csv'
disease_path = base_path / 'diseases.csv'

# Load data and drop the lesser represented diseases
disease_data = pd.read_csv(disease_path)
metadata = pd.read_csv(metadata_path)

disease_data = disease_data.drop('LK', axis=1)
disease_data = disease_data.drop('SL', axis=1)
disease_data = disease_data.drop('CAM', axis=1)
disease_data = disease_data.drop('AMP', axis=1)


#initialise the labels and attention mask dictionaries
encoded_data = dict.fromkeys(metadata.columns.to_list() + ['disease'])
for key in encoded_data.keys():
    encoded_data[key] = []

attention_data = dict.fromkeys(['image','0','1','2','3','4'])
for key in attention_data.keys():
    attention_data[key] = []

# Iterate through all disease and metadata, converting each value to a integer and building attention masks for each row
# Attention masks are used to define which cells have data for each label and which cells don't, important for the metdata classifier and
# diffusion prompt
for disease_row, metadata_row in zip(disease_data.iterrows(), metadata.iterrows()):
    i = disease_row[0]
    disease_row = disease_row[1]
    metadata_row = metadata_row[1]

    disease, attn1 = disease_to_int(disease_row)
    sex, attn2 = sex_to_int(metadata_row['sex'])
    age, attn3 = age_to_int(metadata_row['age'])
    site, attn4 = anatom_site_to_int(metadata_row['anatom_site'])
    bm, attn5 = benign_malignant_to_int(metadata_row['benign_malignant'])

    attention_data['image'].append(metadata_row['image'])
    attention_data['0'].append(attn1)
    attention_data['1'].append(attn2)
    attention_data['2'].append(attn3)
    attention_data['3'].append(attn4)
    attention_data['4'].append(attn5)

    encoded_data['image'].append(metadata_row['image'])
    encoded_data['disease'].append(disease)
    encoded_data['sex'].append(sex)
    encoded_data['age'].append(age)
    encoded_data['anatom_site'].append(site)
    encoded_data['benign_malignant'].append(bm)

for key in encoded_data.keys():
    print(key, len(encoded_data[key]))

# Save all data 
encoded_data = pd.DataFrame(encoded_data)

attention_data = pd.DataFrame(attention_data)

samples = attention_data[['0', '1', '2', '3', '4']].sum(axis=1) > 0
encoded_data = encoded_data[samples]
attention_data = attention_data[samples]

encoded_data.to_csv('all_encoded.csv', index=False)
attention_data.to_csv('all_attention.csv', index=False)























# encoded_metadata = dict.fromkeys(metadata.columns)
# for key in encoded_metadata.keys():
#     encoded_metadata[key] = np.empty((len(metadata,)))

# encoded_metadata['image'] = metadata['image']
# for i in range(len(metadata)):
#     encoded_metadata['sex'][i] = 0 if metadata['sex'].iloc[i] == 'male' else 1
#     encoded_metadata['age'][i] = int(metadata['age'].iloc[i]//5) if not math.isnan(metadata['age'].iloc[i]) else metadata['age'].iloc[i]
#     encoded_metadata['anatom_site'][i] = anatom_sites.get(metadata['anatom_site'].iloc[i], np.nan)
#     encoded_metadata['benign_malignant'][i] = metadata['benign_malignant'].iloc[i]

# encoded_metadata = pd.DataFrame(encoded_metadata)
# encoded_metadata.to_csv('./e_metadata.csv', index=False)