from models.disease_classifier import DiseaseTraining
from utils.datasets import Classification

import torch
torch.set_float32_matmul_precision('high')

from torch.utils.data import DataLoader

import lightning as pl
from argparse import ArgumentParser
from pathlib import Path
import os
from torch.utils.data import random_split


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='The path to the ISIC datset')
    parser.add_argument('-s', '--save_dir', type=str, help='The path to save results')
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train with')
    parser.add_argument('-l', '--learning_rate', type=float, help='The learning rate of the optimiser')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    save_dir = Path(args.save_dir)

    dataset = Classification(root=Path(args.path))

    traindata, validdata, testdata = random_split(dataset, [0.7, 0.1, 0.2])

    trainloader = DataLoader(traindata, batch_size=16, shuffle=True, num_workers=12)
    validloader = DataLoader(validdata, batch_size=16, shuffle=False, num_workers=12)
    testloader = DataLoader(testdata, batch_size=16, shuffle=False, num_workers=12)

    classifier = DiseaseTraining(lr=args.learning_rate)

    trainer = pl.Trainer(max_epochs=args.epochs)
    trainer.fit(classifier, train_dataloaders=trainloader, val_dataloaders=validloader)
    trainer.test(classifier, testloader)

    torch.save(classifier.model.state_dict(), save_dir/'final_diseae_model.pt')