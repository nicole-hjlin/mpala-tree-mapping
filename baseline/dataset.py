from os import path
from torch.utils.data import Dataset
import laspy

BASE_DIR = path.dirname(path.abspath(__file__))

class MpalaTreeLiDAR(Dataset):
    def __init__(self, data_path, ids, labels, classes, transform=None):
        super().__init__()
        self.transform = transform
        self.n = len(ids)
        self.x = []
        self.x_trans = {}
        self.y = labels
        self.classes = classes
        
        for id in ids:
            f = path.join(data_path, f'treeID_{id}.las')
            self.x.append(laspy.read(f))

    def __getitem__(self, i):
        if self.transform and i in self.x_trans:
            x = self.x_trans[i]
        elif self.transform:
            x = self.x[i]
            x = self.transform(x)
            self.x_trans[i] = x
        else:
            x = self.x[i]
        y = self.y[i]

        return x, y

    def __len__(self):
        return self.n

    def randomize(self):
        pass
