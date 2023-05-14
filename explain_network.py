from models.disease_classifier import DiseaseClassifier
from models.datasets import Classification
import torch
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image

from pathlib import Path

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

reverse_normalise = transforms.Compose([
    transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])
])

# topil = transforms.To()

base = Path('E:\\Programming\\Datasets\\All_ISIC\\real')

dataset = Classification(root=base, transform=transform)

model = DiseaseClassifier()
model.load_state_dict(torch.load('fake_model.pt'))

input_tensor, target = dataset[3335]
input_tensor = input_tensor.unsqueeze(dim=0)
# target = torch.tensor([target]).unsqueeze(dim=0)
print(input_tensor.shape)

targets = [ClassifierOutputTarget(target)]

cam = GradCAM(model, [model.classifier.layer4[-1]], use_cuda=False)

cam_mask = cam(input_tensor=input_tensor, targets=targets)
cam_mask = cam_mask[0, :]

image = reverse_normalise(input_tensor)
image = image.reshape((3, 224, 224))
image = image.permute((1,2,0))
print(image.shape)
image = image.numpy()

masked_image = show_cam_on_image(image, cam_mask, use_rgb=True, image_weight=0.6)

masked_image = Image.fromarray(masked_image)
masked_image.save('CAM_Grad_Fake.jpg')

