from PIL import Image
from pathlib import Path
import os

base = Path('E:\Programming\Datasets\All_ISIC')
original = base / 'images'

if not os.path.exists('E:\\Programming\\Datasets\\All_ISIC\\resized_images'):
    os.mkdir('E:\\Programming\\Datasets\\All_ISIC\\resized_images')

resized = base / 'resized_images'

files = os.listdir(original)
# print(files)

for file in files:
    orig_file = original / file
    new_file = resized / file

    image = Image.open(orig_file)
    aspect_ratio = image.height / image.width

    if image.height < image.width:
        new_height = 512
        new_width = int(512*image.width / image.height)
    else:
        new_width = 512
        new_height = int(512*image.height/image.width)
    image = image.resize((new_width, new_height))
    image.save(new_file)