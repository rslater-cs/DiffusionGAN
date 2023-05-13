import csv
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset
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
from datasets import Classification, stratified_split_indexes
from pathlib import Path
import seaborn as sns

WORKERS = 0

DATASET_PATH = Path('E:\\Programming\\Datasets\\All_ISIC')
dataset_classification = Classification(root=DATASET_PATH)

#visualize the dataset
disease_labels = dataset_classification.disease_labels
label_occurences = dict()

#zero the count for each label
for i in range(10):
    label_occurences[i] = 0

#get the number of images for each label
for label in disease_labels:
    label_occurences[label] += 1

plt.figure()
plt.bar(x=list(map(str, label_occurences.keys())), height=list(label_occurences.values()))
plt.show()

#By visualizing the number of images for each labels, we can see that the number of images for the labels 4, 6, 11, 12 are
#so little, so we will drop the images that belong to those labels and we will create a classsification model for 10 labels

# current_path = os.getcwd()
# folder_path = os.path.join(current_path, 'models')

# #create train subpath and validation sabpath
# train_subpath = os.path.join(folder_path, 'traincls')
# validation_subpath = os.path.join(folder_path, 'valcls')

# if os.path.exists(train_subpath):
#     shutil.rmtree(train_subpath)
# os.mkdir(train_subpath)

# if os.path.exists(validation_subpath):
#     shutil.rmtree(validation_subpath)
# os.mkdir(validation_subpath)

# #create 0 to 9 subfolders in train subfolder and validation subfolder
# for i in range(10):
#     createPath = str(i)
#     subpath_to_create_train = os.path.join(train_subpath, createPath)
#     subpath_to_create_val = os.path.join(validation_subpath, createPath)
#     os.mkdir(subpath_to_create_train)
#     os.mkdir(subpath_to_create_val)

# MODEL_LABELS = [0, 1, 2, 3, 5, 7, 8, 9, 10, 13]
# new_model_labels = []
# image_paths = dataset_classification.image_paths

# #for each class I will put 10 percent of the records of that class as validation
# #this variable will hold the number of validation records for each class
# number_train = []
# for label_key, label_value in label_occurences.items():
#     try:
#         new_index_of_label = MODEL_LABELS.index(label_key)
#         number_train_records = int(label_value - (label_value * 0.10))
#         number_train.append(number_train_records)
#     except ValueError:
#         print(f"The image has the label {label_key} and will be dropped later.")

# number_of_records_as_train = dict()

# #zero the count for each label
# for i in range(10):
#     number_of_records_as_train[i] = 0

""""
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
"""

# for each class I will put 70 percent train, 10 percent validation and 20 percent test
train, valid, test = stratified_split_indexes(dataset_classification.disease_labels, splits=[0.7, 0.1, 0.2])

train_dataset = Subset(dataset_classification, train)
valid_dataset = Subset(dataset_classification, valid)
test_dataset = Subset(dataset_classification, test)


# #debug not right
# for index, label in disease_labels.iteritems():
#     try:
#         index_model_new_label = MODEL_LABELS.index(label)
#         number_of_records_as_train[index_model_new_label] += 1
#         new_model_labels.append(index_model_new_label)
#         currentImagePath = image_paths[index]
#         if (number_of_records_as_train[index_model_new_label] <= 16):
#             moveToTrain = os.path.join(train_subpath, str(index_model_new_label))
#             shutil.copy(currentImagePath, moveToTrain)
#         elif ((number_of_records_as_train[index_model_new_label] >= 17) and (number_of_records_as_train[index_model_new_label] <= 32)):
#             moveToVal = os.path.join(validation_subpath, str(index_model_new_label))
#             shutil.copy(currentImagePath, moveToVal)
#     except ValueError:
#         print(f"The image has the label {label} and will be dropped.")



## This code segment has been dropped as classification class has dropped rows as default
# label_occurences_new = dict()
#
# #zero the count for each label
# for i in range(10):
#     label_occurences_new[i] = 0
#
# #get the number of images for each label
# for label in new_model_labels:
#     label_occurences_new[label] += 1
#
# plt.figure()
# plt.bar(x=list(map(str, label_occurences_new.keys())), height=list(label_occurences_new.values()))
# plt.show()


batch_size = 32
#batch_size_debug = 4


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=WORKERS)
val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=WORKERS)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=WORKERS)

#train_dataloader = DataLoader(tensor_resized_image_data_train, batch_size=batch_size_debug , shuffle = True)
#val_dataloader = DataLoader(tensor_resized_image_data_val, batch_size=batch_size_debug, shuffle = True)



device = "cuda" if torch.cuda.is_available() else "cpu"


model = models.resnet101(pretrained=True)

# for param in model.parameters():
#     param.requires_grad = False

f1_macro_score = []
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loss = []
validation_loss = []
train_accuracy = []
validatation_accuracy = []
#epochs = 10
epochs = 4
best_model_lowest_val_loss = -1
best_model_highest_f1_score = -1

predicted_labels = []
actual_labels = []

#train the model
def train(dataloader, val_dataloader, model, loss_fn, optimizer):
    model = model.to(device)
    model.train()
    lowest_val_loss = float('inf')
    highest_f1_score = -0.1
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss_of_epoch = 0
        validation_loss_of_epoch = 0
        train_accuracy_of_epoch = 0
        validation_accuracy_of_epoch = 0
        
        for X, y in tqdm(dataloader):
            optimizer.zero_grad()

            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

            train_loss_of_epoch += loss.item()
            _, predicted = torch.max(pred.data, 1)
            train_accuracy_of_epoch += (predicted == y).sum().item()

        #if batch % 100 == 0:
            #loss, current = loss.item(), batch * len(X)
            #print(f"train_loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
        train_loss_of_epoch = train_loss_of_epoch / len(train_dataset)
        train_accuracy_of_epoch = train_accuracy_of_epoch / len(train_dataset)
        train_loss.append(train_loss_of_epoch)
        train_accuracy.append(train_accuracy_of_epoch)

        predicted_labels_epoch = []
        actual_labels_epoch = []
        for X, y in tqdm(val_dataloader):
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

        #actual_labels.append(actual_labels_epoch)
        #predicted_labels.append(predicted_labels_epoch)
        f1_score_epoch = f1_score(actual_labels_epoch, predicted_labels_epoch, average='macro')
        f1_macro_score.append(f1_score_epoch)

        validation_loss_of_epoch = validation_loss_of_epoch / len(valid_dataset)
        validation_accuracy_of_epoch = validation_accuracy_of_epoch / len(valid_dataset)
        validation_loss.append(validation_loss_of_epoch)
        validatation_accuracy.append(validation_accuracy_of_epoch)

        torch.save(model.state_dict(), f"weights_epoch{t+1}.pt")

        if (validation_loss_of_epoch < lowest_val_loss):
            lowest_val_loss = validation_loss_of_epoch
            best_model_lowest_val_loss = t + 1

        if (f1_score_epoch > highest_f1_score):
            highest_f1_score = f1_score_epoch
            best_model_highest_f1_score = t + 1

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
          .format(epochs+1, epochs, train_loss_of_epoch, train_accuracy_of_epoch, validation_loss_of_epoch, validation_accuracy_of_epoch))
        
    return actual_labels, predicted_labels, best_model_lowest_val_loss, best_model_highest_f1_score


actual_labels, predicted_labels, best_model_lowest_val_loss, best_model_highest_f1_score = train(train_dataloader, val_dataloader, model, loss, optimizer)

print(f"Best model on lowest val score was on epoch {best_model_lowest_val_loss}. If you want to load the weights of this model you should load the the weights_epoch{best_model_lowest_val_loss} pt file")
print(f"Best model on highest f1 score was on epoch {best_model_highest_f1_score}. If you want to load the weights of this model you should load the the weights_epoch{best_model_highest_f1_score} pt file")

#metrics on validation data
#precision_score_for_class = []
#recall_score_for_class = []
#f1_score_for_class = []

#precision_macro_score = []
#recall_macro_score = []


#for j in range(len(actual_labels)):
    #precision_score_for_class.append(precision_score(actual_labels[j], predicted_labels[j], average=None))
    #recall_score_for_class.append(recall_score(actual_labels[j], predicted_labels[j], average=None))
    #f1_score_for_class.append(f1_score(actual_labels[j], predicted_labels[j], average=None))

    #precision_macro_score.append(precision_score(actual_labels[j], predicted_labels[j], average='macro'))
    #recall_macro_score.append(recall_score(actual_labels[j], predicted_labels[j], average='macro'))
    

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
#ndex_of_best_model = best_model-1

#precision score for each class
#precision_for_each_class = precision_score_for_class[index_of_best_model]
#for i, label in enumerate(LABELS):
    #print(f"Label {label} precision score = {round(precision_for_each_class[i], 2)}")

#print(" ")

#recall score for each class
#recall_score_for_each_class = recall_score_for_class[index_of_best_model]
#for i, label in enumerate(LABELS):
    #print(f"Label {label} recall score = {round(recall_score_for_each_class[i], 2)}")

#print(" ")

#f1 score for each class
#f1_score_for_each_class = f1_score_for_class[index_of_best_model]
#for i, label in enumerate(LABELS):
    #print(f"Label {label} f1 score = {round(f1_score_for_each_class[i], 2)}")



#plot the confusion matrix
def plot_confusion_matrix():
    confusion_m = confusion_matrix(actual_labels, predicted_labels)
    confusion_plot = pd.DataFrame(confusion_m, range(10), range(10))
    plt.figure(figsize = (35, 35))
    sns.heatmap(confusion_plot, annot=True)

print("Done!")
