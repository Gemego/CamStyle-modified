import os.path
import torch.utils.data as data
import PIL as pil
import torchvision.transforms as transforms
import random
import os.path as osp
from glob import glob
import re

DATA_ROOT = "D:\\智能计算系统\\Market-1501-v15.09.15\\Market-1501-v15.09.15"
LOAD_SIZE = 286
FINE_SIZE = 256


class ReidDataset(data.Dataset):
    def __init__(self):
        self.serial_batches = True
        self.root = DATA_ROOT
        self.dir = os.path.join(self.root, 'bounding_box_train')

        self.A_paths = self.preprocess(self.dir, cam_id=1)
        self.B_paths = self.preprocess(self.dir, cam_id=2)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform = None
        self.init_transform()

    def init_transform(self):
        transform_list = []
        osize = [LOAD_SIZE, LOAD_SIZE]
        transform_list.append(transforms.Resize(osize, pil.Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(FINE_SIZE))

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def preprocess(self, path, cam_id=1, extra_cam_id=-1):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        ret = []
        fpaths = sorted(glob(osp.join(path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if cam not in [cam_id, extra_cam_id]:
                continue
            ret.append(os.path.join(path, fname))
        return ret

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = pil.Image.open(A_path).convert('RGB')
        B_img = pil.Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)
        input_nc = 3
        output_nc = 3

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)


class CustomDatasetDataLoader(data.Dataset):
    def __init__(self):
        self.dataset = ReidDataset()
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=not self.dataset.serial_batches,
            num_workers=4)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
