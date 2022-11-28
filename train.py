import torch
from torch import nn, optim
from torchmetrics import AUROC
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from typing import Tuple, Dict
from tqdm import tqdm
import wandb
from simpleview.model import SimpleView
from dataset import MpalaTreeLiDAR
import util
import pandas as pd

wandb.login()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(
    project: str,
    config: Dict,
) -> nn.Module:
    with wandb.init(project=project, config=config):
        config = wandb.config
        model, dataset, loader, testloader, crit, opt = make(config)
        print('batch size: ', config.batch_size)
        print('loader size: ', len(loader))
        print('dataset size: ', len(dataset))
        # model = torch.nn.DataParallel(model)
        model = model.to(device)
        wandb.watch(model, crit, log='all', log_freq=10)
        total_batches = len(loader) * config.epochs
        example_ct = 0
        batch_ct = 0
        pbar = tqdm(total=total_batches)
        for epoch in range(config.epochs):
            for x, y in loader:
                example_ct += len(x)
                batch_ct += 1
                x, y = x.to(device), y.to(device)

                y_pred = model(x)
                loss = crit(y_pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                acc = (y_pred.argmax(-1) == y).float().mean()

                if batch_ct % 8 == 0:
                    wandb.log({
                        'epoch': epoch,
                        'loss': loss,
                        'acc': acc,
                    }, step=example_ct)
                pbar.update(1)
            
            with torch.no_grad():
                test_y_pred = torch.Tensor([]).to(device)
                test_y = torch.Tensor([]).to(device)
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    test_y_pred = torch.cat([test_y_pred, y_pred])
                    test_y = torch.cat([test_y, y])
                wandb.log({
                    'test_auc': AUROC(len(dataset.classes))(test_y_pred, test_y.int()),
                    'test_acc': (test_y_pred.argmax(-1) == test_y).float().mean(),
                }, step=example_ct)
    return model


def make(
    config: Dict
) -> Tuple[nn.Module, DataLoader, DataLoader, nn.CrossEntropyLoss, optim.SGD]:
    # Make the dataset
    transform = transforms.Compose([
        util.ToPointCloud(),
        util.ProjectPointCloud(uniform_norm=config.normalize),
    ]) if config.spicy else transforms.Compose([
        util.ToPointCloud(),
        util.ProjectPointCloud(uniform_norm=config.normalize),
        util.ExpandChannels(channels=1),
    ])

    dataset = MpalaTreeLiDAR(
        dir=config.data_dir,
        labels=pd.read_csv(config.label_path),
        min_points=config.min_points,
        top_species=config.top_species,
        transform=transform,
    )

    # Split train and test sets
    sub_train, sub_test = random_split(
        dataset,
        [config.train_split, 1-config.train_split]
    )

    # Make dataloaders
    train_loader = DataLoader(
        sub_train,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        sub_test,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # Make the model
    model = SimpleView(
        num_views=6,
        num_classes=len(dataset.classes),
        expand_projections=not config.spicy
    )

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
    )
    
    return model, dataset, train_loader, test_loader, criterion, optimizer
