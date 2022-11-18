import torch
from torch import nn, optim
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
    with wandb.init(project=project, config=config, mode='offline'):
        config = wandb.config
        model, loader, testloader, crit, opt = make(config)
        wandb.watch(model, crit, log='all', log_freq=10)
        # total_batches = len(loader) * config.epochs
        example_ct = 0
        batch_ct = 0
        for epoch in tqdm(range(config.epochs)):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = crit(y_pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                example_ct += len(x)
                batch_ct += 1
                if ((batch_ct + 1) % 25) == 0:
                    wandb.log({'epoch': epoch, 'loss': loss}, step=example_ct)
        # test(model, testloader)
    return model


def make(
    config: Dict
) -> Tuple[nn.Module, DataLoader, DataLoader, nn.CrossEntropyLoss, optim.SGD]:
    # Make the dataset
    dataset = MpalaTreeLiDAR(
        dir=config.data_dir,
        labels=pd.read_pickle(config.label_path),
        min_points=config.min_points,
        transform=transforms.Compose([
            util.ToPointCloud(),
            util.ProjectPointCloud(),
            util.ExpandChannels(channels=1),
        ]),
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
    )

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
    )
    
    return model, train_loader, test_loader, criterion, optimizer
