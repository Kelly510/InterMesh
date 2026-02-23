import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import copy
from configs.paths import dataset_root
from .base import BASE


class MPII(BASE):

    def __init__(self, split='train', downsample=1, **kwargs):
        super(MPII, self).__init__(**kwargs)
        assert split == 'train'

        self.ds_name = 'mpii'
        self.split = split
        self.dataset_path = os.path.join(dataset_root, 'mpii')
        annots_path = os.path.join(self.dataset_path, 'MPII_NA_SMPL.npz')
        self.annots = np.load(annots_path, allow_pickle=True)['annots'][()]
        if downsample != 1:
            self.annots = {k: v for idx, (k, v) in enumerate(self.annots.items()) if idx % downsample == 0}
        self.img_names = list(self.annots.keys())

    def __len__(self):
        return len(self.img_names)

    def get_raw_data(self, idx):
        img_id = idx % len(self.img_names)
        img_name = self.img_names[img_id]
        annots = copy.deepcopy(self.annots[img_name])
        img_path = os.path.join(self.dataset_path, 'images', img_name)

        pnum = len(annots)
        cam_rot = torch.eye(3, 3).repeat(pnum, 1, 1).float()
        cam_trans = torch.zeros(pnum, 3).float()

        betas_list = []
        poses_list = []
        transl_list = []
        cam_intrinsics_list = []

        for i in range(pnum):
            #smpl and cam
            smpl_param = annots[i]['smpl_param']
            cam_param = annots[i]['cam_param']
            cam_intrinsics = torch.tensor([[cam_param['focal'][0], 0., cam_param['princpt'][0]], [0, cam_param['focal'][1], cam_param['princpt'][1]], [0, 0, 1]])
            betas = torch.tensor(smpl_param['shape'])
            poses = torch.tensor(smpl_param['pose'])
            transl = torch.tensor(smpl_param['trans'])
            betas_list.append(betas)
            poses_list.append(poses)
            transl_list.append(transl)
            cam_intrinsics_list.append(cam_intrinsics)

        betas = torch.stack(betas_list).float()
        poses = torch.stack(poses_list).float()
        transl = torch.stack(transl_list).float()
        cam_intrinsics = torch.stack(cam_intrinsics_list).float()

        raw_data = {'img_path': img_path, 'ds': 'mpii', 'pnum': len(betas), 'betas': betas, 'poses': poses, 'transl': transl, 'cam_intrinsics': cam_intrinsics, 'cam_rot': cam_rot, 'cam_trans': cam_trans, '3d_valid': False, 'detect_all_people': False}

        return raw_data
