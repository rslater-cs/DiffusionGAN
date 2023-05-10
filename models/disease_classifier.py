from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch import optim
from torchvision import models
import lightning as pl
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix


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

        self.precision = MulticlassPrecision(10, average=None)
        self.recall = MulticlassRecall(10, average=None)
        self.f1score = MulticlassF1Score(10, average=None)
        self.confusion_matrix = MulticlassConfusionMatrix(10)

        self.lr = lr

        self.predicted_labels = []
        self.true_labels = []

    def training_step(self, batch_idx, batch):
        image, disease = batch

        pred = self.model(image)
        loss = self.loss_func(pred, disease)

        _, predidx = torch.max(pred.data, 1)
        acc = 100*(predidx == disease).sum()/disease.shape[0]
        self.log("train/accuracy", acc, on_step=False, on_epoch=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch_idx, batch):
        image, disease = batch

        pred = self.model(image)
        loss = self.loss_func(pred, disease)

        _, predidx = torch.max(pred.data, 1)
        acc = 100*(predidx == disease).sum()/disease.shape[0]
        self.log("valid/accuracy", acc, on_step=False, on_epoch=True)
        self.log("valid/loss", loss, on_step=False, on_epoch=True)

        return loss
    
    def test_step(self, batch_idx, batch) -> STEP_OUTPUT | None:
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

        self.predicted_labels.extend(predidx.tolist())
        self.true_labels.extend(disease.tolist())

    def on_test_end(self):
        self.predicted_labels = torch.tensor(self.predicted_labels)
        self.true_labels = torch.tensor(self.true_labels)

        pr = self.precision(self.predicted_labels, self.true_labels)
        self.log_dict({'test/precision':pr})

        rc = self.recall(self.predicted_labels, self.true_labels)
        self.log_dict({'test/recall':rc})

        f1 = self.f1score(self.predicted_labels, self.true_labels)
        self.log_dict({'test/f1':f1})

        cm = self.confusion_matrix(self.predicted_labels, self.true_labels)
        torch.save({'confusion_matrix':cm}, f'{self.logger.root_dir}/confusion_matrix.pt')
        
    
    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.lr)