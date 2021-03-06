import os.path
import torch.utils.data as data
import PIL as pil
import torchvision.transforms as transforms
import random
import os.path as osp
from glob import glob
import re

# DATA_ROOT = "D:\\智能计算系统\\Market-1501-v15.09.15\\Market-1501-v15.09.15"
DATA_ROOT = r"D:\AICS_final\CamStyle-master\CycleGAN-for-CamStyle\data\market"
LOAD_SIZE = 286
FINE_SIZE = 256


class ReidDataset(data.Dataset):
    def __init__(self, camA, camB, isTrain):  # wsy :add isTrain
        self.serial_batches = True
        self.root = DATA_ROOT
        self.dir = os.path.join(self.root, 'bounding_box_train')
        # -------------------wsy------------------------------------------
        # self.A_paths = self.preprocess(self.dir, cam_id=camA)
        # self.B_paths = self.preprocess(self.dir, cam_id=camB)
        self.isTrain = isTrain  # 便于后续其他函数使用
        if self.isTrain:
            self.A_paths = self.preprocess(self.dir, cam_id=camA)
            self.B_paths = self.preprocess(self.dir, cam_id=camB)
        else:
            self.A_paths = self.preprocess(self.dir, cam_id=camA, extra_cam_id=camB)
            self.B_paths = self.preprocess(self.dir, cam_id=camA, extra_cam_id=camB)
        # ----------------------------------------------------------------
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform = None
        self.init_transform()

    def init_transform(self):
        transform_list = []
        osize = [LOAD_SIZE, LOAD_SIZE]
        transform_list.append(transforms.Resize(osize, pil.Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(FINE_SIZE))

        # ---------wsy---------------------这里与text无关，但好像与train有关，可以看看是否是加漏了
        # if opt.isTrain and not opt.no_flip:
        #     transform_list.append(transforms.RandomHorizontalFlip())
        # -----------------------------------
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


class DatasetDataLoader(data.Dataset):
    def __init__(self, camA=1, camB=2, isTrain=True):  # wsy :此处设置为train,则train.py不用改
        self.dataset = ReidDataset(camA, camB, isTrain)  # wsy
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
        for _, _data in enumerate(self.dataloader):
            yield _data
