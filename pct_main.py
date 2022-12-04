from __future__ import print_function
from dataset import MpalaTreeLiDARToPCT, MpalaTreeLiDAR
import time
from PCT_Pytorch.model import Pct
from PCT_Pytorch.util import cal_loss, IOStream
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
import sklearn.metrics as metrics
import utils
import wandb
import pickle
from torchmetrics import AUROC

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' +
              args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' +
              args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' +
              args.exp_name + '/' + 'data.py.backup')
    wandb.login()
    wandb.init(project=args.exp_name, config=args)


def train(args, io):
    train, test = load_tree_lidar_data(args)
    train_loader = DataLoader(
        train, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test, num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda:0" if args.cuda else "cpu")

    model = Pct(args).to(device)
    print('--------- model: ', str(model))
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100,
                        momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = cal_loss
    best_test_acc = 0

    wandb.watch(model, criterion, log='all', log_freq=10)

    for epoch in range(args.epochs):
        scheduler.step()
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        idx = 0
        total_time = 0.0
        for data, label in (train_loader):
            data = data.float()
            # print('---------- train data: ', data)
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            # print('---------- data size: ', data.size())
            opt.zero_grad()

            start_time = time.time()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1

        print('train total time is', total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        print('----- train true:', train_true)
        print('----- train pred:', train_pred)

        # fpr, tpr, _ = metrics.roc_curve(preds, logits)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        results = {
            'epoch': epoch,
            'train loss': train_loss*1.0/count,
            'train acc': metrics.accuracy_score(train_true, train_pred)
            # 'Train Avg Acc': metrics.balanced_accuracy_score(train_true, train_pred)
            # "pr": wandb.plot.pr_curve(train_true, train_pred, labels=None, classes_to_plot=None)
            # 'test_auc': AUROC(len(train.classes))(test_y_pred, train_pred)),
        }

        wandb.log(results)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        for data, label in test_loader:
            data = data.float()
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            start_time = time.time()
            logits = model(data)
            end_time = time.time()
            total_time += (end_time - start_time)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        print('test total time is', total_time)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        print('----- test true:', test_true)
        print('----- test pred:', test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(
            test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)

        with torch.no_grad():
            test_y_pred = torch.Tensor([]).to(device)
            test_y = torch.Tensor([]).to(device)
            for x, y in test_loader:
                x = x.float()
                x, y = x.to(device), y.to(device)
                x = x.permute(0, 2, 1)
                y_pred = model(x)
                test_y_pred = torch.cat([test_y_pred, y_pred])
                test_y = torch.cat([test_y, y])
            wandb.log({
                'epoch': epoch,
                'test_auc': AUROC(len(test.dataset.classes))(test_y_pred, test_y.int()),
                'test_acc': (test_y_pred.argmax(-1) == test_y).float().mean(),
                'test_loss': test_loss*1.0/count
            })

        # results = {
        #     'Test Loss': test_loss*1.0/count,
        #     'Test Acc': test_acc,
        #     'Test Avg Acc': avg_per_class_acc,
        #     "pr": wandb.plot.pr_curve(train_true, train_pred, labels=None, classes_to_plot=None)
        # }

        # wandb.log(results)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            wandb.log({'best_test_acc': best_test_acc})
            torch.save(model.state_dict(),
                       'checkpoints/%s/models/model.t7' % args.exp_name)

        if epoch == args.nepoch - 1:
            current_path = os.path.abspath(__file__)
            with open(os.path.join(current_path, 'pct_preds.pickle'), 'wb') as f:
                pickle.dump(test_pred, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(current_path, 'labels.pickle'), 'wb') as f:
                pickle.dump(test_true, f, protocol=pickle.HIGHEST_PROTOCOL)


def test(args, io):
    _, test = load_tree_lidar_data(args)
    test_loader = DataLoader(
        test, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct(args).to(device)
    model = nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []

    for data, label in test_loader:
        data = data.float()
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        preds = logits.max(dim=1)[1]
        if args.test_batch_size == 1:
            test_true.append([label.cpu().numpy()])
            test_pred.append([preds.detach().cpu().numpy()])
        else:
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

    # Plot AUC and compare the two
    # fpr, tpr, _ = sklearn.metrics.roc_curve(y_test,  y_pred_proba)
    # auc = sklearn.metrics.roc_auc_score(y_test, y_pred_proba)
    # plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (
        test_acc, avg_per_class_acc)
    io.cprint(outstr)


def load_tree_lidar_data(args):
    dataset = MpalaTreeLiDAR(
        dir=args.data_dir,
        labels=pd.read_csv(args.label_path),
        min_points=args.min_points,
        top_species=args.top_species,
        transform=transforms.Compose([
            utils.ToPointCloud(),
        ]),
    )
    # TODO: Fix test/train split process, remove need to initialize partition
    dataset = MpalaTreeLiDARToPCT(dataset, args.num_points, 'test')

    # Split train and test sets
    sub_train, sub_test = random_split(
        dataset,
        [args.train_split, 1-args.train_split]
    )

    sub_train.partition = 'train'
    return sub_train, sub_test


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    # parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
    #                     choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='../PCT_Pytorch/checkpoints/best/models/model.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--data_dir', type=str,
                        default="../data", help='Path to LiDAR tree dataset')
    parser.add_argument('--label_path', type=str,
                        default="../labels.csv", help='Path to LiDAR labels')
    parser.add_argument('--min_points', type=int,
                        default=500, help='Minimum number of points in a tree LiDAR scan file')
    parser.add_argument('--train_split', type=int,
                        default=0.8, help='Ratio of training data to test data')
    parser.add_argument('--output_channels', type=int,
                        default=41, help='Number of classes')
    parser.add_argument('--top_species', type=int, help='')

    args = parser.parse_args()

    _init_(args)

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
