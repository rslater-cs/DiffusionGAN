from models.disease_classifier import DiseaseTraining
from utils.datasets import Classification
from torch.utils.data import DataLoader
import torch
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
    datalen = len(dataset)
    lengths = [int(0.7*datalen), int(0.1*datalen), int(0.2*datalen)]

    if sum(lengths) < datalen:
        lengths[-1] += datalen-sum(lengths)

    traindata, validdata, testdata = random_split(dataset, lengths)

    trainloader = DataLoader(traindata, batch_size=32, shuffle=True, num_workers=12)
    validloader = DataLoader(validdata, batch_size=32, shuffle=True, num_workers=12)
    testloader = DataLoader(testdata, batch_size=32, shuffle=True, num_workers=12)

    classifier = DiseaseTraining(lr=args.learning_rate)

    print(classifier)

    trainer = pl.Trainer(max_epochs=args.epochs, default_root_dir=save_dir)
    trainer.fit(classifier, train_dataloaders=trainloader, validloader=validloader, testloader=testloader)

    torch.save(classifier.model.state_dict(), save_dir/'final_model.pt')