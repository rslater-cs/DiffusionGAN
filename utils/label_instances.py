from pathlib import Path
from datasets import Classification
import pandas as pd

base = Path('./')

dataset = Classification(base)

labels: pd.DataFrame = dataset.disease_labels

print(labels.value_counts().sort_index())


