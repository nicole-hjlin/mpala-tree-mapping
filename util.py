from transforms3d.euler import euler2mat
import torch
import numpy as np
import pandas as pd
from math import radians
import imageio
from typing import Tuple, List


def load_dataset_metainfo(
    min_points: int,
    max_points: int = 1e9,
    desc_path: str = 'data_desc.pkl',
) -> Tuple[List[int], List[int], List[str]]:
    '''
    Returns
    ids : List[int]
        tree IDs, e.g. [41020, 41021, ...]
    labels : List[int]
        ground truth labels, e.g. [0, 1, 9, 2, ...]
    classes : List[str]
        tree species, e.g. ['Euclea divinorum', 'Acacia etbaica', ...]
    '''
    df = pd.read_pickle(desc_path)
    print(f'Found {len(df)} trees in description dataframe')
    df = df[df['num_total'].between(min_points, max_points)]
    print(f'Only {len(df)} trees with points in the range [{min_points}, {max_points}]')
    ids = df['tree_id'].to_list()
    labels = df['Label']
    classes = labels.unique().tolist()
    labels = [classes.index(l) for l in labels]
    print(ids)
    return ids, labels, classes


def las_to_pc(las, hide_class_10: bool = False):
    mask = las.classification != (10 if hide_class_10 else -1)
    x = (las.points.X * las.header.scale[0] + las.header.offset[0])[mask]
    y = (las.points.Y * las.header.scale[1] + las.header.offset[1])[mask]
    z = (las.points.Z * las.header.scale[2] + las.header.offset[2])[mask]
    pc = torch.tensor(np.array([x, y, z]).T, dtype=float)
    classes = torch.tensor(las.classification[mask], dtype=int)
    return pc, classes


def rot_m(xrot, yrot, zrot):
    return torch.tensor(euler2mat(radians(zrot), radians(yrot), radians(xrot)))


def project_point_cloud(pc, xrot=0, yrot=90, zrot=0, width=224, scale=0.5, s=2, darkmode=True, uniform_norm=False):
    image = torch.zeros(width, width)
    pc @= rot_m(zrot, yrot, xrot)
    pc -= pc.mean(dim=0)
    pc /= 7 if uniform_norm else pc.norm(dim=-1).max()

    zorder = (-pc[:,2]).argsort()
    pc = pc[zorder,:]

    pc[:,2] -= pc[:,2].min()
    pc[:,2] /= pc[:,2].max()
    pc[:,2] = 1 - pc[:,2]
    
    for x, y, z in pc:
        fn = lambda x: width * (scale * x + 1/2)
        x = fn(x).round().int()
        y = fn(y).round().int()
        try:
            if s == 1:
                image[x,y] = z
            else:
                l, h = int(s/2), round(s/2)
                image[x-l:x+h, y-l:y+h] = z
        except IndexError:
            continue

    return image if darkmode else 1 - image


def sexy_gif(pc, path: str, **kwargs):
    imageio.mimsave(path, [
        (255*project_point_cloud(pc, xrot=xrot, **kwargs)).byte()
        for xrot in torch.linspace(0, 360, 10)
    ] + [
        (255*project_point_cloud(pc, yrot=yrot, **kwargs)).byte()
        for yrot in torch.linspace(90, 450, 10)
    ])


default_views = [
    (0, 0, 0),
    (0, 180, 0),
    (0, 90, 0),
    (90, 90, 0),
    (180, 90, 0),
    (270, 90, 0),
]


class ToPointCloud:
    def __init__(
        self,
        hide_class_10: bool = False,
    ):
        self.hide_class_10 = hide_class_10

    def __call__(self, las):
        pc, _ = las_to_pc(las, self.hide_class_10)
        return pc


class ProjectPointCloud:
    def __init__(
        self,
        uniform_norm: bool,
        views: List[Tuple[float, float, float]] = default_views,
    ):
        self.uniform_norm = uniform_norm
        self.views = views

    def __call__(self, x):
        proj = []
        for xrot, yrot, zrot in self.views:
            proj.append(project_point_cloud(
                x,
                xrot=xrot,
                yrot=yrot,
                zrot=zrot,
                uniform_norm=self.uniform_norm,
            ))
        proj = torch.stack(proj)
        return proj


class ExpandChannels:
    def __init__(
        self,
        channels: int = 3,
    ):
        self.channels = channels

    def __call__(self, x):
        x = x.unsqueeze(dim=-3)
        x = x.repeat((1, self.channels, 1, 1))
        return x
