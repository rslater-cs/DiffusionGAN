from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.optim import Adam
from torchvision import models
import lightning as pl

# This is no longer used but is good to refer to for knowing different head lengths
head_lengths = {
    'disease':10,
    'sex':2,
    'age':19,
    'site':9,
    'malignant_benign':2
}

class MetadataClassifier(nn.Module):
    def __init__(self, dropout=0.5) -> None:
        super().__init__()

        #Using ConvNext tiny as a backbone with pretrained weights (to speed up training)

        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights, dropout=dropout)

        total_output_length = 42

        # output is the total length of all output nodes (disease, age, sex, site e.t.c) which are seperated later on
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm((768,)),
            nn.Linear(768, total_output_length)
        )

    def forward(self, x: torch.Tensor):
        # output: B 42
        x = self.backbone(x)

        # seperate the outputs into the correct classes
        disease, sex, age, site, mb = x[:,:10], x[:,10:12], x[:,12:31], x[:,31:40], x[:,40:]
        
        return disease, sex, age, site, mb
    
# This is a pytorch lightning module, it allows us to define a training loop very simply
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
        
        # Calculate the cross entropy loss for all output heads
        for i in range(len(outputs)):
            losses[i] = self.loss_func(outputs[i], labels[i])

        # change losses from (H, B) to (B, H) shape where H is the number of heads (5)
        losses = losses.transpose(dim0=0, dim1=1)
        
        # Use attention mask to zero out the heads that do not have a label 
        losses = losses * attn_mask

        # Calculate the mean loss of only the heads that have labels
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
        # Using adam optimiser
        return Adam(self.classifier.parameters(), lr=self.lr)




        

        

