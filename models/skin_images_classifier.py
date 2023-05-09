import csv
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
import shutil
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
#import seaborn as sns


resize_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

# Classification dataset and metdata dataset for classification and dgan systems
class Classification(Dataset):
    def __init__(self, transform: transforms.Compose = resize_transform) -> None:
        super().__init__()
        self.transform = transform

        current_path = os.getcwd()
        folder_path = os.path.join(current_path, 'models')
        data_path = os.path.join(folder_path,'all_encoded.csv')
        attn_path = os.path.join(folder_path,'all_attention.csv')

        labels = pd.read_csv(data_path)
        attn = pd.read_csv(attn_path)

        image_names = labels['image']
        disease_labels = labels['disease']
        disease_attn = attn['0']

        samples = disease_attn == 1

        image_names = image_names[samples]

        images_skin_path = os.path.join(current_path, 'All_ISIC')

        for i in range(len(image_names)):
            image_names.iloc[i] = os.path.join(images_skin_path, image_names.iloc[i])
            #image_names.iloc[i] = Path(root / 'images' / image_names.iloc[i])
        
        self.image_paths = image_names
        self.disease_labels = disease_labels[samples]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = self.transform(image)

        disease = self.disease_labels.iloc[index]

        return image, disease 


dataset_classification = Classification()

#visualize the dataset
disease_labels = dataset_classification.disease_labels
label_occurences = dict()

#zero the count for each label
for i in range(14):
    label_occurences[i] = 0

#get the number of images for each label
for label in disease_labels:
    label_occurences[label] += 1

plt.figure()
plt.bar(x=list(map(str, label_occurences.keys())), height=list(label_occurences.values()))
plt.show()

#By visualizing the number of images for each labels, we can see that the number of images for the labels 4, 6, 11, 12 are
#so little, so we will drop the images that belong to those labels and we will create a classsification model for 10 labels

current_path = os.getcwd()
folder_path = os.path.join(current_path, 'models')

#create train subpath and validation sabpath
train_subpath = os.path.join(folder_path, 'traincls')
validation_subpath = os.path.join(folder_path, 'valcls')

if os.path.exists(train_subpath):
    shutil.rmtree(train_subpath)
os.mkdir(train_subpath)

if os.path.exists(validation_subpath):
    shutil.rmtree(validation_subpath)
os.mkdir(validation_subpath)

#create 0 to 9 subfolders in train subfolder and validation subfolder
for i in range(10):
    createPath = str(i)
    subpath_to_create_train = os.path.join(train_subpath, createPath)
    subpath_to_create_val = os.path.join(validation_subpath, createPath)
    os.mkdir(subpath_to_create_train)
    os.mkdir(subpath_to_create_val)

MODEL_LABELS = [0, 1, 2, 3, 5, 7, 8, 9, 10, 13]
new_model_labels = []
image_paths = dataset_classification.image_paths

#for each class I will put 10 percent of the records of that class as validation
#this variable will hold the number of validation records for each class
number_train = []
for label_key, label_value in label_occurences.items():
    try:
        new_index_of_label = MODEL_LABELS.index(label_key)
        number_train_records = int(label_value - (label_value * 0.10))
        number_train.append(number_train_records)
    except ValueError:
        print(f"The image has the label {label_key} and will be dropped later.")

number_of_records_as_train = dict()

#zero the count for each label
for i in range(10):
    number_of_records_as_train[i] = 0
for index, label in disease_labels.iteritems():
    try:
        index_model_new_label = MODEL_LABELS.index(label)
        number_of_records_as_train[index_model_new_label] += 1
        new_model_labels.append(index_model_new_label)
        currentImagePath = image_paths[index]
        if (number_of_records_as_train[index_model_new_label] < number_train[index_model_new_label]):
            moveToTrain = os.path.join(train_subpath, str(index_model_new_label))
            shutil.copy(currentImagePath, moveToTrain)
        else:
            moveToVal = os.path.join(validation_subpath, str(index_model_new_label))
            shutil.copy(currentImagePath, moveToVal)
    except ValueError:
        print(f"The image has the label {label} and will be dropped.")


#debug not right
#for index, label in disease_labels.iteritems():
#    try:
#        index_model_new_label = MODEL_LABELS.index(label)
#        number_of_records_as_train[index_model_new_label] += 1
#        new_model_labels.append(index_model_new_label)
#        currentImagePath = image_paths[index]
#        if (number_of_records_as_train[index_model_new_label] <= 16):
#            moveToTrain = os.path.join(train_subpath, str(index_model_new_label))
#            shutil.copy(currentImagePath, moveToTrain)
#        elif ((number_of_records_as_train[index_model_new_label] >= 17) and (number_of_records_as_train[index_model_new_label] <= 32)):
#            moveToVal = os.path.join(validation_subpath, str(index_model_new_label))
#            shutil.copy(currentImagePath, moveToVal)
#    except ValueError:
#        print(f"The image has the label {label} and will be dropped.")




label_occurences_new = dict()

#zero the count for each label
for i in range(10):
    label_occurences_new[i] = 0

#get the number of images for each label
for label in new_model_labels:
    label_occurences_new[label] += 1

plt.figure()
plt.bar(x=list(map(str, label_occurences_new.keys())), height=list(label_occurences_new.values()))
plt.show()

#set the new labels to the Classification object
dataset_classification.disease_labels = new_model_labels

tensor_resized_image_data_train = datasets.ImageFolder(root=train_subpath, transform = resize_transform)
tensor_resized_image_data_val = datasets.ImageFolder(root=validation_subpath, transform = resize_transform)


batch_size = 64
#batch_size_debug = 4

train_dataloader = DataLoader(tensor_resized_image_data_train, batch_size=batch_size , shuffle = True)
val_dataloader = DataLoader(tensor_resized_image_data_val, batch_size=batch_size, shuffle = True)

#train_dataloader = DataLoader(tensor_resized_image_data_train, batch_size=batch_size_debug , shuffle = True)
#val_dataloader = DataLoader(tensor_resized_image_data_val, batch_size=batch_size_debug, shuffle = True)



device = "cuda" if torch.cuda.is_available() else "cpu"


model = models.resnet101(pretrained=True)

for param in model.parameters():
    param.requires_grad = False


num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loss = []
validation_loss = []
train_accuracy = []
validatation_accuracy = []
epochs = 18
#epochs = 3


predicted_labels = []
actual_labels = []

#train the model
def train(dataloader, val_dataloader, model, loss_fn, optimizer):
    model = model.to(device)
    model.train()
    for t in tqdm(range(epochs)):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss_of_epoch = 0
        validation_loss_of_epoch = 0
        train_accuracy_of_epoch = 0
        validation_accuracy_of_epoch = 0

        for batch, (X, y) in tqdm(enumerate(dataloader)):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_of_epoch += loss.item()
            _, predicted = torch.max(pred.data, 1)
            train_accuracy_of_epoch += (predicted == y).sum().item()

        #if batch % 100 == 0:
            #loss, current = loss.item(), batch * len(X)
            #print(f"train_loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
        train_loss_of_epoch = train_loss_of_epoch / len(dataloader)
        train_accuracy_of_epoch = train_accuracy_of_epoch / len(dataloader)
        train_loss.append(train_loss_of_epoch)
        train_accuracy.append(train_accuracy_of_epoch)

        predicted_labels_epoch = []
        actual_labels_epoch = []
        for batch, (X, y) in tqdm(enumerate(val_dataloader)):
            X, y = X.to(device), y.to(device)
            model.eval()

            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred, y)
            
            validation_loss_of_epoch = validation_loss_of_epoch + loss.item()
            _, predicted = torch.max(pred.data, 1)

            predicted_labels_epoch = np.concatenate((predicted_labels_epoch, predicted.cpu().numpy())).astype(int)
            actual_labels_epoch = np.concatenate((actual_labels_epoch, y.cpu().numpy())).astype(int)

            validation_accuracy_of_epoch += (predicted == y).sum().item()

            #if batch % 100 == 0:
                #loss, current = loss.item(), batch * len(X)
                #print(f"val_loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        actual_labels.append(actual_labels_epoch)
        predicted_labels.append(predicted_labels_epoch)

        validation_loss_of_epoch = validation_loss_of_epoch / len(val_dataloader)
        validation_accuracy_of_epoch = validation_accuracy_of_epoch / len(val_dataloader)
        validation_loss.append(validation_loss_of_epoch)
        validatation_accuracy.append(validation_accuracy_of_epoch)

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
          .format(epochs+1, epochs, train_loss_of_epoch, train_accuracy_of_epoch, validation_loss_of_epoch, validation_accuracy_of_epoch))
        
    return actual_labels, predicted_labels


actual_labels, predicted_labels = train(train_dataloader, val_dataloader, model, loss, optimizer)

print("Done!")

#metrics on validation data
precision_score_for_class = []
recall_score_for_class = []
f1_score_for_class = []

precision_macro_score = []
recall_macro_score = []
f1_macro_score = []

for j in range(len(actual_labels)):
    precision_score_for_class.append(precision_score(actual_labels[j], predicted_labels[j], average=None))
    recall_score_for_class.append(recall_score(actual_labels[j], predicted_labels[j], average=None))
    f1_score_for_class.append(f1_score(actual_labels[j], predicted_labels[j], average=None))

    precision_macro_score.append(precision_score(actual_labels[j], predicted_labels[j], average='macro'))
    recall_macro_score.append(recall_score(actual_labels[j], predicted_labels[j], average='macro'))
    f1_macro_score.append(f1_score(actual_labels[j], predicted_labels[j], average='macro'))

LABELS = ['MEL', 'NV', 'BCC', 'SL', 'LK', 'DF', 'VASC', 'SCC', 'AMP', 'UNK']

x_axis_epochs = []
for z in range(epochs):
    x_axis_epochs.append(z+1)

#graph for training and validation loss
plt.plot(x_axis_epochs, train_loss, 'g', label='Training loss')
plt.plot(x_axis_epochs, validation_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
#graph for training and validation accuracy
plt.plot(x_axis_epochs, train_accuracy, 'g', label='Training accuracy')
plt.plot(x_axis_epochs, validatation_accuracy, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
#graph for f1 score macro
plt.plot(x_axis_epochs, f1_macro_score, 'g', label='f1 macro score for each epoch on val data')
plt.title('f1 macro score')
plt.xlabel('Epochs')
plt.ylabel('f1 macro score')
plt.legend()
plt.show()

#table for precision TODO
#a, b = plt.subplots()
#table = b.table(cellText=precision_score_for_class[len(precision_score_for_class)-1], colLabels=None, cellLoc='center', loc='center')

#table for recall TODO


#plot the confusion matrix
confusion_m = confusion_matrix(actual_labels[len(actual_labels)-1], predicted_labels[len(predicted_labels)-1])
confusion_plot = pd.DataFrame(confusion_m, range(10), range(10))
plt.figure(figsize = (35, 35))
#sns.heatmap(confusion_plot, annot=True)
