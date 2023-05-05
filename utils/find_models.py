from torchvision import models

cl_models = models.list_models()

for model in cl_models:
    print(model)