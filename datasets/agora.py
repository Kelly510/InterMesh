import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from configs.paths import dataset_root
import copy
from tqdm import tqdm
from .base import BASE


class AGORA(BASE):

    def __init__(self, split='train', downsample=1, **kwargs):
        super(AGORA, self).__init__(**kwargs)
        assert split in ['train', 'test', 'validation']
        assert downsample == 1

        self.ds_name = 'agora'
        self.split = split
        self.dataset_path = os.path.join(dataset_root, 'agora')

        # no annotations are available for AGORA-test
        if split == 'test':
            self.mode = 'infer'
            self.img_names = os.listdir(os.path.join(self.dataset_path, self.split))
        else:
            if self.split == 'train':
                annots_path = os.path.join(self.dataset_path, 'smpl_neutral_annots', 'annots_smpl_{}_fit.npz'.format(split))
            else:
                annots_path = os.path.join(self.dataset_path, 'smpl_neutral_annots', 'annots_smpl_{}.npz'.format(split))
            self.annots = np.load(annots_path, allow_pickle=True)['annots'][()]
            self.img_names = list(self.annots.keys())

    def __len__(self):
        return len(self.img_names)

    def get_raw_data(self, idx):
        img_id = idx % len(self.img_names)
        img_name = self.img_names[img_id]

        if self.mode == 'infer':
            img_path = os.path.join(self.dataset_path, self.split, img_name)
            raw_data = {'img_path': img_path, 'img_name': img_name, 'ds': 'agora'}
            return raw_data

        annots = copy.deepcopy(self.annots[img_name])
        img_path = os.path.join(self.dataset_path, self.split, img_name)

        valid_mask = np.where(annots['isValid'])[0]

        # this should not happen
        if len(valid_mask) == 0:
            print(img_name, 'lack valid person')
            exit(0)

        cam_intrinsics = torch.from_numpy(np.array(annots['cam_intrinsics']))
        cam_rot = torch.from_numpy(np.array(annots['cam_rot'])[valid_mask])
        cam_trans = torch.from_numpy(np.array(annots['cam_trans'])[valid_mask])

        betas_list = []
        poses_list = []
        transl_list = []

        kid = []

        for pNum in range(len(annots['isValid'])):
            if not annots['isValid'][pNum]:
                continue

            gt = annots['smpl_gt'][pNum]
            betas = gt['betas'].flatten()[:10]
            betas_list.append(torch.from_numpy(betas))
            full_poses = torch.cat([torch.from_numpy(gt['global_orient'].flatten()), torch.from_numpy(gt['body_pose'].flatten())])
            poses_list.append(full_poses)
            transl_list.append(torch.from_numpy(gt['transl'].flatten()))

            kid.append(annots['kid'][pNum])

        betas = torch.stack(betas_list)
        poses = torch.stack(poses_list)
        transl = torch.stack(transl_list)

        raw_data = {'img_path': img_path, 'ds': 'agora', 'pnum': len(betas), 'betas': betas.float(), 'poses': poses.float(), 'transl': transl.float(), 'kid': torch.tensor(kid), 'cam_rot': cam_rot.float(), 'cam_trans': cam_trans.float(), 'cam_intrinsics': cam_intrinsics.float(), '3d_valid': True, 'detect_all_people': True}

        return raw_data
