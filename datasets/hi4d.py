import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import copy
from configs.paths import dataset_root
from .base import BASE


class Hi4D(BASE):

    def __init__(self, split='train', downsample=1, **kwargs):
        super(Hi4D, self).__init__(**kwargs)
        assert split in ['train', 'val', 'test']

        self.ds_name = 'hi4d'
        self.split = split
        self.dataset_path = os.path.join(dataset_root, 'hi4d')
        self.annots_path = os.path.join(self.dataset_path, 'hi4d_smpl_{}.npz'.format(split))
        self.annots = np.load(self.annots_path, allow_pickle=True)['annots'].item()
        if downsample != 1:
            self.annots = {k: v for idx, (k, v) in enumerate(self.annots.items()) if idx % downsample == 0}
        self.img_names = list(self.annots.keys())

    def __len__(self):
        return len(self.img_names)

    def get_raw_data(self, idx):
        img_id = idx % len(self.img_names)
        img_name = self.img_names[img_id]

        annots = copy.deepcopy(self.annots[img_name])
        img_path = os.path.join(self.dataset_path, img_name)

        pnum = len(annots)
        betas_list = []
        poses_list = []
        transl_list = []
        cam_rot_list = []
        cam_trans_list = []
        cam_intrinsics_list = []

        for i in range(pnum):
            betas = torch.from_numpy(annots[i]['betas'])
            poses = torch.from_numpy(np.concatenate([annots[i]['global_orient'], annots[i]['body_pose']], axis=0))
            transl = torch.from_numpy(annots[i]['transl'])
            cam_rot = torch.from_numpy(annots[i]['cam_rot'])
            cam_trans = torch.from_numpy(annots[i]['cam_trans']).reshape(3)
            cam_intrinsics = torch.tensor([[annots[i]['focal'][0], 0, annots[i]['princpt'][0]], [0, annots[i]['focal'][1], annots[i]['princpt'][1]], [0, 0, 1]])
            betas_list.append(betas)
            poses_list.append(poses)
            transl_list.append(transl)
            cam_rot_list.append(cam_rot)
            cam_trans_list.append(cam_trans)
            cam_intrinsics_list.append(cam_intrinsics)

        betas = torch.stack(betas_list).float()
        poses = torch.stack(poses_list).float()
        transl = torch.stack(transl_list).float()
        cam_rot = torch.stack(cam_rot_list).float()
        cam_trans = torch.stack(cam_trans_list).float()  # [pnum, 3]
        cam_intrinsics = torch.stack(cam_intrinsics_list).float()

        raw_data = {'img_path': img_path, 'ds': 'harmony4d', 'pnum': len(betas), 'betas': betas, 'poses': poses, 'transl': transl, 'cam_intrinsics': cam_intrinsics, 'cam_rot': cam_rot, 'cam_trans': cam_trans, '3d_valid': True, 'detect_all_people': True}

        return raw_data
