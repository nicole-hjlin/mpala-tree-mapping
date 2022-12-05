from os import path
from torch.utils.data import Dataset
import laspy
import argparse
import glob
import pandas as pd
import re
import utils
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class MpalaTreeLiDAR(Dataset):
    def __init__(
        self,
        dir: str,
        labels: pd.DataFrame,
        min_points: int,
        top_species: int,
        transform: transforms.Compose,
    ):
        self.x, self.y = [], []
        datapoints = {}
        for filepath in glob.glob(path.join(dir, "treeID_*.las")):
            id = int(re.findall("treeID_(.+)\.las$", filepath)[0])
            # print('Processing', id)
            if id not in list(labels['tree_id']):
                continue
            las = laspy.read(filepath)
            if len(las.classification) < min_points:
                continue
            datapoints[id] = las

        labels = labels[labels['tree_id'].isin(datapoints.keys())]

        if top_species:
            print(
                f'Classifying only the top {top_species} species. The rest are considered as a single class "OTHER"')
            top_species = labels['label'].value_counts()[
                :top_species].index.to_list()
            labels.loc[~labels['label'].isin(top_species), 'label'] = 'OTHER'
            print(labels['label'].value_counts())

        self.classes = labels['label'].unique().tolist()
        def id_to_label(
            x): return labels[labels['tree_id'] == x]['label'].item()

        for id, las in datapoints.items():
            self.y.append(self.classes.index(id_to_label(id)))
            self.x.append(las)

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
        num_points: int = 1024,  # Uniform number of points that are taken by PCT
        partition: str = 'train',
    ):
        self.data = data
        self.num_points = num_points
        self.partition = partition
        self.classes = data.classes

    def __getitem__(self, item):
        x, y = self.data.__getitem__(item)
        if self.partition == 'train':
            x = random_point_dropout(x)
            x = translate_pointcloud(x)
            # Dangerous action?
            np.random.shuffle(x.numpy())
        return x[:self.num_points], y

    def __len__(self):
        return self.data.__len__()


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(
        pointcloud, xyz1), xyz2).float()
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default="./data", help='')
    parser.add_argument('--label_path', type=str,
                        default="./labels.csv", help='')
    parser.add_argument('--min_points', type=int, default=500, help='')
    parser.add_argument('--num_points', type=int, default=1024, help='')
    parser.add_argument('--use_baseline', type=bool, default=False, help='')

    config = parser.parse_args()

    if config.use_baseline:
        transform_config = transforms.Compose([
            utils.ToPointCloud(),
            utils.ProjectPointCloud(),
            utils.ExpandChannels(channels=1),
        ])
    else:
        transform_config = transforms.Compose([
            utils.ToPointCloud(),
        ])

    dataset = MpalaTreeLiDAR(
        dir=config.data_dir,
        labels=pd.read_csv(config.label_path),
        min_points=config.min_points,
        transform=transform_config,
    )

    if not config.use_baseline:
        dataset = MpalaTreeLiDARToPCT(
            dataset, num_points=config.num_points, partition='train')

    for i in dataset:
        print(i[0].shape)
        print(i[1])
