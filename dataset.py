from os import path
from torch.utils.data import Dataset
import laspy
import argparse
import glob
import pandas as pd
import re
import util
import numpy as np
from torchvision import transforms
from PCT_Pytorch.data import random_point_dropout, jitter_pointcloud, translate_pointcloud


class MpalaTreeLiDAR(Dataset):
    def __init__(
        self,
        dir: str,
        labels: pd.DataFrame,
        min_points: int,
        transform: transforms.Compose,
    ):
        self.classes = labels['label'].unique().tolist()
        print('Classes', len(self.classes))
        def encode_label(x): return self.classes.index(x)
        def id_to_label(
            x): return labels[labels['tree_id'] == x]['label'].item()

        self.x, self.y = [], []

        print('id:', labels['tree_id'][0])
        for filepath in glob.glob(path.join(dir, "treeID_*.las")):
            id = int(re.findall("treeID_(.+)\.las$", filepath)[0])
            # print('Processing', id)
            if id not in list(labels['tree_id']):
                continue
            las = laspy.read(filepath)
            if len(las.classification) < min_points:
                continue
            print(id, 'has num points', len(las.classification))
            self.x.append(las)
            self.y.append(encode_label(id_to_label(id)))

        self.n = len(self.x)
        self.transformed = set()
        self.transform = transform

    def __getitem__(self, i):
        x = self.x[i]
        y = self.y[i]

        if i not in self.transformed:
            x = self.transform(x)
            self.x[i] = x
            self.transformed.add(i)

        return x, y

    def __len__(self):
        return self.n

    def randomize(self):
        pass

# Used as a wrapper class to feed into PCT Model


class MpalaTreeLiDARToPCT(Dataset):
    def __init__(
        self,
        data: MpalaTreeLiDAR,
        num_points: int = 1024,
        # data_path: str = None,
        partition: str = 'train',
    ):
        self.data = data
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        x, y = self.data.__getitem__(item)
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud)
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return x[:self.num_points], y

    def __len__(self):
        return self.data.__len__()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default="./data", help='')
    parser.add_argument('--label_path', type=str,
                        default="./labels.csv", help='')
    parser.add_argument('--min_points', type=int, default=500, help='')
    parser.add_argument('--use_baseline', type=bool, default=True, help='')

    config = parser.parse_args()

    if config.use_baseline:
        transform_config = transforms.Compose([
            util.ToPointCloud(),
            util.ProjectPointCloud(),
            util.ExpandChannels(channels=1),
        ])
    else:
        transform_config = transforms.Compose([
            util.ToPointCloud(),
        ])

    dataset = MpalaTreeLiDAR(
        dir=config.data_dir,
        labels=pd.read_csv(config.label_path),
        min_points=config.min_points,
        transform=transform_config,
    )
