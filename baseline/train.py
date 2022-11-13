# https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb#scrollTo=CRdLxMS_RTtk
import torch
from torch import nn, optim
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision import transforms
from typing import Union, Dict
import wandb
from typing import Tuple

from models.simpleview import SimpleView

wandb.login()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import pc_transform
from dataset import MpalaTreeLiDAR
from util import load_fake_dataset

def model_pipeline(
    project: str,
    config: Dict,
) -> nn.Module:
    # tell wandb to get started
    with wandb.init(project=project, config=config):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config)

      # and use them to train the model
      train(model, train_loader, criterion, optimizer, config)

      # and test its final performance
      test(model, test_loader)

    return model

def make(
    config: Dict
) -> Tuple[nn.Module, DataLoader, DataLoader, nn.CrossEntropyLoss, optim.SGD]:
    # Make the data
    dataset, sub_train, sub_test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(sub_train, batch_size=config.batch_size)
    test_loader = make_loader(sub_test, batch_size=config.batch_size)

    # Make the model
    model = SimpleView(
        num_views=6,
        num_classes=len(dataset.classes),
        architecture=config.architecture,
    )

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    
    return model, train_loader, test_loader, criterion, optimizer

def get_data(
    min_points: int,
    max_points: int,
    train_split: float,
) -> Tuple[Dataset, Subset, Subset]:
    # Get information about our data, namely tree IDs with appropriate number of points
    ids, labels, classes = load_fake_dataset(
        min_points=min_points,
        max_points=max_points,
    )

    dataset = MpalaTreeLiDAR(
        data_path='../data/MpalaForestGEO_LasClippedtoTreePolygons',
        ids=ids,
        labels=labels,
        classes=classes,
        transform=transforms.Compose([
            pc_transform.ToPointCloud(),
            pc_transform.ProjectPointCloud(),
            pc_transform.ExpandChannels(channels=1),
        ]),
    )
    sub_train, sub_test = random_split(dataset, [train_split, 1-train_split])
    
    return dataset, sub_train, sub_test


def make_loader(
    dataset: Dataset,
    batch_size: int,
) -> DataLoader:
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    return loader


def train(
    model: nn.Module,
    loader: DataLoader,
    criterion: DataLoader,
    optimizer: nn.CrossEntropyLoss,
    config: optim.SGD,
) -> None:
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
    
            # Forward pass ➡
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass ⬅
            optimizer.zero_grad()
            loss.backward()

            # Step with optimizer
            optimizer.step()

            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

