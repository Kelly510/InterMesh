import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import copy
import json
from configs.paths import dataset_root
from .base import BASE


class CHI3D(BASE):

    def __init__(self, split='train', downsample=1, **kwargs):
        super(CHI3D, self).__init__(**kwargs)
        assert split in ['train', 'test']

        self.ds_name = 'chi3d'
        self.split = split
        self.dataset_path = os.path.join(dataset_root, 'chi3d')
        if split == 'test':
            self.img_names = []
            self.mode = 'infer'
            with open(os.path.join(self.dataset_path, 'chi3d_template.json'), 'r') as f:
                result_template = json.load(f)
            for subj in os.listdir(os.path.join(self.dataset_path, self.split)):
                for action in os.listdir(os.path.join(self.dataset_path, self.split, subj, 'images')):
                    for fname in os.listdir(os.path.join(self.dataset_path, self.split, subj, 'images', action)):
                        frame_id = int(fname[:-4])
                        if frame_id not in result_template[subj][action]['other']['video_fr_ids']:
                            continue
                        self.img_names.append(os.path.join(subj, 'images', action, fname))
        else:
            self.annots_path = os.path.join(self.dataset_path, 'chi3d_smpl_{}.npz'.format(split))
            self.annots = np.load(self.annots_path, allow_pickle=True)['annots'].item()
            if downsample != 1:
                self.annots = {k: v for idx, (k, v) in enumerate(self.annots.items()) if idx % downsample == 0}
            self.img_names = list(self.annots.keys())

    def __len__(self):
        return len(self.img_names)

    def get_raw_data(self, idx):
        img_id = idx % len(self.img_names)
        img_name = self.img_names[img_id]

        if self.mode == 'infer':
            img_path = os.path.join(self.dataset_path, self.split, img_name)
            raw_data = {'img_path': img_path, 'img_name': img_name, 'ds': 'fit3d'}
            return raw_data

        annots = copy.deepcopy(self.annots[img_name])
        img_path = os.path.join(self.dataset_path, self.split, img_name)

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

        raw_data = {'img_path': img_path, 'ds': 'chi3d', 'pnum': len(betas), 'betas': betas, 'poses': poses, 'transl': transl, 'cam_intrinsics': cam_intrinsics, 'cam_rot': cam_rot, 'cam_trans': cam_trans, '3d_valid': True, 'detect_all_people': True}

        return raw_data
