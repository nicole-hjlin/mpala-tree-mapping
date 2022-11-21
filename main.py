import argparse
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='mpala-lidar-baseline', help='')
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--momentum', type=float, default=0.9, help='')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--architecture', type=str, default='resnet18_4', help='')
parser.add_argument('--min_points', type=int, default=500, help='')
parser.add_argument('--train_split', type=float, default=0.85, help='')
parser.add_argument('--data_dir', type=str, default="./MpalaForestGEO_LasClippedtoTreePolygons", help='')
parser.add_argument('--label_path', type=str, default="./labels.csv", help='')

if __name__ == '__main__':
    config = parser.parse_args()
    train(config.name, config)