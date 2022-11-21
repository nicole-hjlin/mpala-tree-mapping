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
        top_species: int,
        transform: transforms.Compose,
    ):
        self.x, self.y = [], []
        datapoints = {}
        for filepath in glob.glob(path.join(dir, "treeID_*.las")):
            id = int(re.findall("treeID_(.+)\.las$", filepath)[0])
            if id not in list(labels['tree_id']):
                continue
            las = laspy.read(filepath)
            if len(las.classification) < min_points:
                continue
            datapoints[id] = las

        labels = labels[labels['tree_id'].isin(datapoints.keys())]

        if top_species:
            print(f'Classifying only the top {top_species} species. The rest are considered as a single class "OTHER"')
            top_species = labels['label'].value_counts()[:top_species].index.to_list()
            labels.loc[~labels['label'].isin(top_species),'label'] = 'OTHER'
            print(labels['label'].value_counts())

        self.classes = labels['label'].unique().tolist()
        id_to_label = lambda x: labels[labels['tree_id'] == x]['label'].item()
        
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
