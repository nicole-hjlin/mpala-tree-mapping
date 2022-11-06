from os import path
from torch.utils.data import Dataset

BASE_DIR = path.dirname(path.abspath(__file__))

class MpalaTreePointCloud(Dataset):
    def __init__(self, data_path, examples, transforms=None):
        super().__init__()

        self.transforms = transforms

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(BASE_DIR, f))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)
        self.set_num_points(num_points)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = min(self.points.shape[1], pts)

    def randomize(self):
        pass
