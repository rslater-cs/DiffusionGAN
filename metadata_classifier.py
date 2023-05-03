from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch import nn

class MetaClassifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.effnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.effnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, output_size)
        )

# You need to merge all ISIC datasets into one, make sure to keep datasets with metadata seperate as they will be used for this classifier, duplicates need to be checked for