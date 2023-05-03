from torchvision import models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

cl_models = models.list_models()

for model in cl_models:
    print(model)

effnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

print(effnet)