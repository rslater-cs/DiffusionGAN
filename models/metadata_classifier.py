from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.optim import Adam
from torchvision import models
import lightning as pl

# Classifier with heads for different pieces of metadata: disease, age, sex, site, malig/benin
head_lengths = {
    'disease':14,
    'sex':2,
    'age':19,
    'site':9,
    'malignant_benign':2
}

class MetadataClassifier(nn.Module):
    def __init__(self, dropout=0.5) -> None:
        super().__init__()

        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights, dropout=dropout)

        total_output_length = 46

        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm((768,)),
            nn.Linear(768, total_output_length)
        )

    def forward(self, x: torch.Tensor):
        # output: B 46
        x = self.backbone(x)

        disease, sex, age, site, mb = x[:,:14], x[:,14:16], x[:,16:35], x[:,35:44], x[:,44:]
        
        return disease, sex, age, site, mb
    
class MetadataTraining(pl.LightningModule):

    def __init__(self, dropout: float = 0.5, lr: float = 1e-3) -> None:
        super().__init__()

        self.classifier = MetadataClassifier(dropout)
        self.loss_func = nn.CrossEntropyLoss(reduce=False)

        self.lr = lr

    def forward_pass(self, batch, batch_idx):
        x, labels, attn_mask = batch[0], batch[1:6], batch[6]

        B = x.shape[0]

        self.classifier.zero_grad()

        # output: disease_out, sex_out, age_out, site_out, mb_out
        outputs = self.classifier(x)

        losses = torch.empty((5, B), device=self.device)
        
        for i in range(len(outputs)):
            losses[i] = self.loss_func(outputs[i], labels[i])

        losses = losses.transpose(dim0=0, dim1=1)
        
        losses = losses * attn_mask

        loss = torch.sum(losses) / torch.sum(attn_mask)

        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.forward_pass(batch, batch_idx)

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        loss = self.forward_pass(batch, batch_idx)

        self.log("valid/loss", loss)

        return loss
    
    def configure_optimizers(self) -> Any:
        return Adam(self.classifier.parameters(), lr=self.lr)




        

        

