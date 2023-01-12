import argparse
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,
                    default='mpala-lidar-baseline', help='Name of experiment')
parser.add_argument('--learning_rate', type=float, default=0.001, help='SGD learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--epochs', type=int, default=100, help='Number of epcohs')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size')
parser.add_argument('--spicy', action='store_true', help='Treat projections as channels of a single datapoint')
parser.add_argument('--normalize', action='store_true', help='Equally normalize projections across dataset')
parser.add_argument('--min_points', type=int, default=500, help='Minimum number of points')
parser.add_argument('--train_split', type=float, default=0.85, help='Fraction of data to keep as train split')
parser.add_argument('--data_dir', type=str,
                    default="./MpalaForestGEO_LasClippedtoTreePolygons", help='Path to LiDAR data directory')
parser.add_argument('--label_path', type=str, default="./labels.csv", help='Path to labels')
parser.add_argument('--top_species', type=int, help='Number of species to classify. The most frequent species are selected and the rest are considered OTHER')
parser.add_argument('--use_baseline', type=bool, default=True, help='Use the baseline model')

if __name__ == '__main__':
    config = parser.parse_args()
    # TODO: Add logic for calling the baseline
    train(config.name, config)
