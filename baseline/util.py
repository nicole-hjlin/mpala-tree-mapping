from transforms3d.euler import euler2mat
import torch
import numpy as np
import pandas as pd
from math import radians
import imageio
from typing import Tuple, List

def load_fake_dataset(
    min_points: int,
    max_points: int = 1e9,
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
    df = pd.read_pickle('data_desc.pkl')
    print(f'Found {len(df)} trees')
    df = df[df['num_total'].between(min_points, max_points)]
    print(f'Found only {len(df)} trees with points in range [{{min_points}}, {{max_points}}]')
    ids = df['tree_id'].to_list()
    labels = df['Label']
    classes = labels.unique().tolist()
    labels = [classes.index(l) for l in labels]
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

def project_point_cloud(pc, xrot=0, yrot=90, zrot=0, width=224, scale=0.5, s=2):
    image = torch.zeros(width, width)
    pc @= rot_m(zrot, yrot, xrot)
    pc -= pc.mean(dim=0)
    pc /= pc.norm(dim=-1).max()

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

    return image

def sexy_gif(pc, path: str, **kwargs):
    imageio.mimsave(path, [
        (255*project_point_cloud(pc, xrot=xrot, **kwargs)).byte()
        for xrot in torch.linspace(0, 360, 60)
    ] + [
        (255*project_point_cloud(pc, yrot=yrot, **kwargs)).byte()
        for yrot in torch.linspace(90, 450, 60)
    ])
