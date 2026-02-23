import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import copy
from configs.paths import dataset_root
from .base import BASE


class PANOPTIC(BASE):

    def __init__(self, split='test', **kwargs):
        super(PANOPTIC, self).__init__(**kwargs)
        assert split == 'test'

        self.ds_name = 'cmu_panoptic'
        self.split = split
        self.dataset_path = os.path.join(dataset_root, 'cmu_panoptic')
        annots_path = os.path.join(self.dataset_path, 'annots_test.npz')
        self.annots = np.load(annots_path, allow_pickle=True)['annots'][()]
        self.img_names = list(self.annots.keys())

    def __len__(self):
        return len(self.img_names)

    def process_data(self, img, raw_data, rot=0., flip=False, scale=1., crop=False):
        meta_data = copy.deepcopy(raw_data)
        img, M = self.process_img(img, meta_data)
        return img, meta_data, M

    def get_raw_data(self, idx):
        img_id = idx % len(self.img_names)
        img_name = self.img_names[img_id]
        annots = copy.deepcopy(self.annots[img_name])
        img_path = os.path.join(self.dataset_path, 'images', img_name)

        j3ds = torch.tensor(annots['kpts3d'])
        j2ds = torch.tensor(annots['kpts2d'])

        width = annots['width']
        height = annots['height']
        vis = (j2ds[..., 0] > 0) & (j2ds[..., 0] < width) & (j2ds[..., 1] > 0) & (j2ds[..., 1] < height)

        raw_data = {
            'img_path': img_path,
            'ds': 'cmu_panoptic',
            'j3ds': j3ds,
            'j2ds': j2ds[..., :2],
            'vis': vis.float(),
        }

        return raw_data
