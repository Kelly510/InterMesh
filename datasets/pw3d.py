import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import copy
from configs.paths import dataset_root
from .base import BASE


class PW3D(BASE):

    def __init__(self, split='train', downsample=1, **kwargs):
        super(PW3D, self).__init__(**kwargs)
        assert split in ['train', 'test']
        assert downsample == 1

        self.ds_name = '3dpw'
        self.split = split
        self.dataset_path = os.path.join(dataset_root, '3dpw')
        annots_path = os.path.join(self.dataset_path, 'annots_smpl_{}_genders.npz'.format(split))
        self.annots = np.load(annots_path, allow_pickle=True)['annots'][()]
        self.img_names = list(self.annots.keys())

    def __len__(self):
        return len(self.img_names)

    def get_raw_data(self, idx):
        img_id = idx % len(self.img_names)
        img_name = self.img_names[img_id]
        annots = copy.deepcopy(self.annots[img_name])
        img_path = os.path.join(self.dataset_path, img_name)

        pnum = len(annots['betas'])

        cam_intrinsics = torch.tensor(annots['cam_intrinsics']).float().unsqueeze(0)
        cam_rot = torch.tensor(annots['cam_rot']).repeat(pnum, 1, 1).float()
        cam_trans = torch.tensor(annots['cam_trans']).repeat(pnum, 1).float()

        betas = annots['betas']
        poses = torch.cat([annots['global_orient'], annots['body_pose']], dim=1)
        transl = annots['transl']

        genders = annots['genders'] if 'genders' in annots else annots['gender']
        genders = ['female' if gender.lower() in ['f', 'female'] else 'male' for gender in genders]

        raw_data = {'img_path': img_path, 'ds': '3dpw', 'pnum': len(betas), 'betas': betas, 'poses': poses, 'transl': transl, 'cam_rot': cam_rot, 'cam_trans': cam_trans, 'cam_intrinsics': cam_intrinsics, '3d_valid': True, 'genders': genders, 'detect_all_people': False}

        return raw_data
