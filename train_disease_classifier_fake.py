from models.disease_classifier import DiseaseTraining
from models.datasets import Classification, ClassificationFake, stratified_split_indexes

import torch
torch.set_float32_matmul_precision('high')

from torch.utils.data import DataLoader, Subset, ConcatDataset

import lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
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

    dataset = Classification(root=Path(args.path)/'real')

    real_count = dataset.disease_labels.value_counts().sort_index()

    train, valid, test = stratified_split_indexes(dataset.disease_labels, [0.7, 0.1, 0.2])

    traindata = Subset(dataset, train)
    validdata = Subset(dataset, valid)
    testdata = Subset(dataset, test)

    fakedata = ClassificationFake(root=Path(args.path)/'fake')

    fake_count = fakedata.disease_labels.value_counts().sort_index()

    total_count = fake_count + real_count
    total_count = total_count.fillna(0)
    total_count[total_count == 0] = real_count[total_count==0]
    total_count = total_count.astype(int)

    plt.bar(list(range(10)), list(total_count.values))
    plt.show()

    traindata = ConcatDataset([traindata, fakedata])

    trainloader = DataLoader(traindata, batch_size=32, shuffle=True, num_workers=0)
    validloader = DataLoader(validdata, batch_size=32, shuffle=False, num_workers=0)
    testloader = DataLoader(testdata, batch_size=32, shuffle=False, num_workers=0)

    classifier = DiseaseTraining(epochs=args.epochs, lr=args.learning_rate)

    trainer = pl.Trainer(max_epochs=args.epochs)
    trainer.fit(classifier, train_dataloaders=trainloader, val_dataloaders=validloader)
    tst = trainer.test(classifier, testloader)

    print(tst)

    # data = dict.fromkeys(['precision','recall','f1'])
    # data['precision'] = pr
    # data['recall'] = rc
    # data['f1'] = f1

    # plt.matshow(cm)

    # pd.DataFrame(data).to_csv(save_dir/'final_metrics.csv')

    torch.save(classifier.model.state_dict(), save_dir/'final_diseae_model.pt')