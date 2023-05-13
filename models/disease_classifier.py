from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch import optim
from torchvision import models
import lightning as pl
import pandas as pd
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix, MulticlassAccuracy
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class DiseaseClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = models.resnet101(pretrained=True)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, 10)

    def forward(self, x: torch.Tensor):
        return self.classifier(x)
    
class DiseaseTraining(pl.LightningModule):
    def __init__(self, epochs, lr: float = 1e-3) -> None:
        super().__init__()

        self.model = DiseaseClassifier()
        self.loss_func = nn.CrossEntropyLoss()

        self.precision = MulticlassPrecision(10, average=None).to(self.device)
        self.recall = MulticlassRecall(10, average=None).to(self.device)
        self.f1score = MulticlassF1Score(10, average=None).to(self.device)
        self.confusion_matrix = MulticlassConfusionMatrix(10).to(self.device)
        #self.accuracy = MulticlassAccuracy(10)

        self.lr = lr
        self.epochs = epochs

        self.predicted_labels_highest_f1_torch = []
        self.predicted_labels_lowest_loss_torch = []
        self.true_labels_torch = []

        self.predicted_labels_lowest_loss = []
        self.predicted_labels_highest_f1 = []

        self.true_labels = []

        self.trained_at_least_one_epoch = False
        self.train_loss = []
        self.validation_loss = []
        self.train_accuracy = []
        self.validation_accuracy = []

        self.number_train_records = 0
        self.number_val_records = 0

        self.train_accuracy_of_epoch = 0
        self.train_loss_of_epoch = 0
        self.length_of_train_dataloader = 0

        self.validation_accuracy_of_epoch = 0
        self.validation_loss_of_epoch = 0
        self.length_of_val_dataloader = 0

        self.val_predicted_labels_epoch = []
        self.val_actual_labels_epoch = []

        self.val_f1_score_macro_score = []
        self.val_epoch = 0

        self.lowest_val_loss = float('inf')
        self.highest_f1_score = -0.1
        self.best_model_lowest_val_loss = -1
        self.best_model_highest_f1_score = -1


    def training_step(self, batch, batch_idx):
        image, disease = batch

        pred = self.model(image)
        loss = self.loss_func(pred, disease)

        _, predicted = torch.max(pred.data, 1)

        if (batch_idx == 0):
            self.train_accuracy_of_epoch = 0
            self.train_loss_of_epoch = 0
            self.length_of_train_dataloader = 0
            self.number_train_records = 0

        self.train_accuracy_of_epoch += (predicted == disease).sum().item()
        self.train_loss_of_epoch += loss.item()
        self.length_of_train_dataloader += 1

        self.number_train_records += len(disease)

        self.log("train/accuracy", self.train_accuracy_of_epoch/self.number_train_records)
        self.log("train/loss", self.train_loss_of_epoch/self.length_of_train_dataloader)

        self.trained_at_least_one_epoch = True

        return loss
    
    def on_train_epoch_end(self):
        average_train_loss_of_epoch = round((self.train_loss_of_epoch / self.length_of_train_dataloader), 2)
        average_train_accuracy_of_epoch = round(((self.train_accuracy_of_epoch * 100) / self.number_train_records), 2)
        self.train_loss.append(average_train_loss_of_epoch)
        self.train_accuracy.append(average_train_accuracy_of_epoch)

    def validation_step(self, batch, batch_idx):
        image, disease = batch

        pred = self.model(image)
        loss = self.loss_func(pred, disease)

        _, predicted = torch.max(pred.data, 1)


        if (self.trained_at_least_one_epoch == True):
            if (batch_idx == 0):
                self.validation_loss_of_epoch = 0
                self.validation_accuracy_of_epoch = 0
                self.length_of_val_dataloader = 0
                self.number_val_records = 0
                self.val_predicted_labels_epoch = []
                self.val_actual_labels_epoch = []

            self.validation_loss_of_epoch += loss.item()
            self.validation_accuracy_of_epoch += (predicted == disease).sum().item()
            self.length_of_val_dataloader += 1

            self.number_val_records += len(disease)

            self.log("valid/accuracy", self.validation_accuracy_of_epoch/self.number_val_records)
            self.log("valid/loss", self.validation_loss_of_epoch/self.length_of_val_dataloader)

            self.val_predicted_labels_epoch = np.concatenate((self.val_predicted_labels_epoch, predicted.cpu().numpy())).astype(int)
            self.val_actual_labels_epoch = np.concatenate((self.val_actual_labels_epoch, disease.cpu().numpy())).astype(int)

        return loss
    
    def on_validation_epoch_end(self):
        if (self.trained_at_least_one_epoch == True):
            f1_score_epoch = round((f1_score(self.val_actual_labels_epoch, self.val_predicted_labels_epoch, average='macro')), 2)
            self.val_f1_score_macro_score.append(f1_score_epoch)

            average_validation_loss_of_epoch = round((self.validation_loss_of_epoch / self.length_of_val_dataloader), 2)
            average_validation_accuracy_of_epoch = round(((self.validation_accuracy_of_epoch * 100) / self.number_val_records), 2)
            self.validation_loss.append(average_validation_loss_of_epoch)
            self.validation_accuracy.append(average_validation_accuracy_of_epoch)

            self.val_epoch += 1

            if (average_validation_loss_of_epoch < self.lowest_val_loss):
                self.lowest_val_loss = average_validation_loss_of_epoch
                self.best_model_lowest_val_loss = self.val_epoch 
                torch.save(self.model.state_dict(), f"weights_epoch{self.val_epoch}.pt")

            if (f1_score_epoch > self.highest_f1_score):
                self.highest_f1_score = f1_score_epoch
                self.best_model_highest_f1_score = self.val_epoch
                torch.save(self.model.state_dict(), f"weights_epoch{self.val_epoch}.pt")


    #def test_step(self, batch, batch_idx) -> STEP_OUTPU None:
    def test_step(self, batch, batch_idx):

        image, disease = batch

        print(f"Best model on lowest val score was on epoch {self.best_model_lowest_val_loss}. If you want to load the weights of this model you should load the the weights_epoch{self.best_model_lowest_val_loss} pt file")
        
        #Load the model of the epoch that has the lowest validation loss at the end of the epoch
        model_weights_lowest_val_loss = f"weights_epoch{self.best_model_lowest_val_loss}.pt"
        self.model.load_state_dict(torch.load(model_weights_lowest_val_loss))
        pred_lowest_val_loss = self.model(image)
        _, predicted_lowest_val_loss = torch.max(pred_lowest_val_loss.data, 1)

        print(f"Best model on highest f1 score was on epoch {self.best_model_highest_f1_score}. If you want to load the weights of this model you should load the the weights_epoch{self.best_model_highest_f1_score} pt file")
        #Load the model of the epoch that has the highest f1 score at the end of the epoch
        model_weights_highest_f1_score = f"weights_epoch{self.best_model_highest_f1_score}.pt"
        self.model.load_state_dict(torch.load(model_weights_highest_f1_score))
        pred_highest_f1 = self.model(image)
        _, predicted_highest_f1 = torch.max(pred_highest_f1.data, 1)

        self.true_labels.extend(disease.tolist())
        self.predicted_labels_lowest_loss.extend(predicted_lowest_val_loss.tolist())
        self.predicted_labels_highest_f1.extend(predicted_highest_f1.tolist())

        self.predicted_labels_highest_f1_torch = torch.tensor(self.predicted_labels_highest_f1).to(self.device)
        self.predicted_labels_lowest_loss_torch = torch.tensor(self.predicted_labels_lowest_loss).to(self.device)
        self.true_labels_torch = torch.tensor(self.true_labels).to(self.device)

    

    def on_test_epoch_end(self):

        #log the metrics for lowest val score model
        self.log_save_metrics(self.predicted_labels_lowest_loss_torch, self.true_labels_torch, "lowest_val_loss")

        #log the metrics for highest f1 score model
        self.log_save_metrics(self.predicted_labels_highest_f1_torch, self.true_labels_torch, "highest_f1_score")

        x_axis_epochs = []
        for z in range(self.epochs):
            x_axis_epochs.append(z+1)

        #graph for training and validation loss
        plt.plot(x_axis_epochs, self.train_loss, 'g', label='Training loss')
        plt.plot(x_axis_epochs, self.validation_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.figure()
        #graph for training and validation accuracy
        plt.plot(x_axis_epochs, self.train_accuracy, 'g', label='Training accuracy')
        plt.plot(x_axis_epochs, self.validation_accuracy, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.figure()
        #graph for f1 score macro
        plt.plot(x_axis_epochs, self.val_f1_score_macro_score, 'g', label='f1 macro score for each epoch on val data')
        plt.title('f1 macro score')
        plt.xlabel('Epochs')
        plt.ylabel('f1 macro score')
        plt.legend()
        plt.show()

        #confusion matrix for lowest val score
        self.plot_confusion_matrix(self.true_labels, self.predicted_labels_lowest_loss)

        #confusion matrix for highest f1 score
        self.plot_confusion_matrix(self.true_labels, self.predicted_labels_highest_f1)
    
    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def log_save_metrics(self, predicted_labels, true_labels, name):
        pr = self.precision(predicted_labels, true_labels)
        rc = self.recall(predicted_labels, true_labels)
        f1 = self.f1score(predicted_labels, true_labels)
        cm = self.confusion_matrix(predicted_labels, true_labels)
        accuracy = (predicted_labels == true_labels).sum().item()
        pd.DataFrame({"test/accuracy": [accuracy]}).to_csv(f'{self.logger.root_dir}/final_accuracy_{name}.csv')
        pd.DataFrame({"test/precision":pr.tolist(),"test/recall":rc.tolist(),"test/f1":f1.tolist()}).to_csv(f'{self.logger.root_dir}/final_metrics_{name}.csv')
        torch.save({'confusion_matrix':cm}, f'{self.logger.root_dir}/confusion_matrix{name}.pt')
    

    #plot the confusion matrix
    def plot_confusion_matrix(actual_labels, predicted_labels):
        LABELS = ['MEL', 'NV', 'BCC', 'SL', 'LK', 'DF', 'VASC', 'SCC', 'AMP', 'UNK']
        #skleaarn y-true first then y-pred after
        confusion_m = confusion_matrix(actual_labels, predicted_labels)
        confusion_plot = pd.DataFrame(confusion_m, index=LABELS, columns=LABELS)
        plt.figure(figsize = (15, 15))
        heatmap = sns.heatmap(confusion_plot, annot=True)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=90)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)