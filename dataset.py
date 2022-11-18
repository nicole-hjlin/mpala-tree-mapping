from os import path
from torch.utils.data import Dataset
import laspy
import glob
import pandas as pd
import re
from torchvision import transforms


class MpalaTreeLiDAR(Dataset):
    def __init__(
        self, 
        dir: str, 
        labels: pd.DataFrame, 
        min_points: int, 
        transform: transforms.Compose,
    ):
        self.classes = labels['label'].unique().tolist()
        print(len(self.classes))
        encode_label = lambda x: self.classes.index(x)
        id_to_label = lambda x: labels[labels['tree_id'] == x]['label'].item()
        
        self.x, self.y = [], []

        for filepath in glob.glob(path.join(dir, "treeID_*.las")):
            id = int(re.findall("treeID_(.+)\.las$", filepath)[0])
            if id not in list(labels['tree_id']):
                continue
            las = laspy.read(filepath)
            if len(las.classification) < min_points:
                continue
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
