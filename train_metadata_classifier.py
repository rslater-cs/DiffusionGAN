from models.metadata_classifier import MetadataTraining
from utils.datasets import Metadata
from torch.utils.data import DataLoader
import torch
import lightning as pl
from argparse import ArgumentParser
from pathlib import Path
import os

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

    dataset = Metadata(root=Path(args.path))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    classifier = MetadataTraining(dropout=0.5, lr=args.learning_rate)

    print(classifier)

    trainer = pl.Trainer(max_epochs=args.epochs, default_root_dir=save_dir)
    trainer.fit(classifier, dataloader)

    torch.save(classifier.classifier.state_dict(), save_dir/'final_model.pt')