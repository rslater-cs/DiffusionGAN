from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch import optim
from torchvision import models
import lightning as pl
import pandas as pd
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix, MulticlassAccuracy


class DiseaseClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = models.resnet101(pretrained=True)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, 10)

    def forward(self, x: torch.Tensor):
        return self.classifier(x)
    
class DiseaseTraining(pl.LightningModule):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()

        self.model = DiseaseClassifier()
        self.loss_func = nn.CrossEntropyLoss()

        self.precision = MulticlassPrecision(10, average=None).to(self.device)
        self.recall = MulticlassRecall(10, average=None).to(self.device)
        self.f1score = MulticlassF1Score(10, average=None).to(self.device)
        self.confusion_matrix = MulticlassConfusionMatrix(10).to(self.device)
        self.accuracy = MulticlassAccuracy(10)

        self.lr = lr

        self.predicted_labels = []
        self.true_labels = []

    def training_step(self, batch, batch_idx):
        image, disease = batch

        pred = self.model(image)
        loss = self.loss_func(pred, disease)

        _, predidx = torch.max(pred.data, 1)
        acc = 100*(predidx == disease).sum()/disease.shape[0]
        self.log("train/accuracy", acc, on_step=False, on_epoch=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, disease = batch

        pred = self.model(image)
        loss = self.loss_func(pred, disease)

        _, predidx = torch.max(pred.data, 1)
        acc = 100*(predidx == disease).sum()/disease.shape[0]
        self.log("valid/accuracy", acc, on_step=False, on_epoch=True)
        self.log("valid/loss", loss, on_step=False, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        if batch_idx == 0:
            self.predicted_labels = []
            self.true_labels = []

        image, disease = batch

        pred = self.model(image)
        loss = self.loss_func(pred, disease)

        _, predidx = torch.max(pred.data, 1)
        acc = 100*(predidx == disease).sum()/disease.shape[0]
        self.log("test/accuracy", acc, on_step=False, on_epoch=True)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        self.accuracy(pred, disease)

        self.predicted_labels.extend(predidx.tolist())
        self.true_labels.extend(disease.tolist())

    def on_test_epoch_end(self):
        self.predicted_labels = torch.tensor(self.predicted_labels).to(self.device)
        self.true_labels = torch.tensor(self.true_labels).to(self.device)

        pr = self.precision(self.predicted_labels, self.true_labels)

        rc = self.recall(self.predicted_labels, self.true_labels)

        f1 = self.f1score(self.predicted_labels, self.true_labels)

        cm = self.confusion_matrix(self.predicted_labels, self.true_labels)

        self.log("test/accuracy", self.accuracy(self.predicted_labels, self.true_labels))
        pd.DataFrame({"test/precision":pr.tolist(),"test/recall":rc.tolist(),"test/f1":f1.tolist()}).to_csv(f'{self.logger.root_dir}/final_metrics.csv')

        torch.save({'confusion_matrix':cm}, f'{self.logger.root_dir}/confusion_matrix.pt')

        return {'precision':pr, 'recall':rc, 'f1':f1, 'matrix':cm}
        
    
    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.lr)