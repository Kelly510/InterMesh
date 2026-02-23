import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import json
from configs.paths import dataset_root
import copy
from .base import BASE


class H36M(BASE):

    def __init__(self, split='train', downsample=1, **kwargs):
        super(H36M, self).__init__(**kwargs)
        assert split == 'train'

        self.ds_name = 'h36m'
        self.split = split
        self.dataset_path = os.path.join(dataset_root, 'h36m')
        annots_path = os.path.join(self.dataset_path, 'annots_smpl_train_small.npz')
        self.annots = np.load(annots_path, allow_pickle=True)['annots'][()]
        if downsample != 1:
            self.annots = {k: v for idx, (k, v) in enumerate(self.annots.items()) if idx % downsample == 0}
        self.img_names = list(self.annots.keys())

        actions_path = os.path.join(self.dataset_path, 'actions.json')
        cameras_path = os.path.join(self.dataset_path, 'cameras.json')

        with open(actions_path) as f:
            self.actions = json.load(f)
        with open(cameras_path) as f:
            self.cameras = json.load(f)

    def __len__(self):
        return len(self.img_names)

    def get_raw_data(self, idx):
        img_id = idx % len(self.img_names)
        img_name_orig = self.img_names[img_id]

        # convert image name to existing files
        parts = os.path.basename(img_name_orig).split('_')
        action = self.actions['_'.join(parts[:6])]
        camera = self.cameras[str(int(parts[7]))]
        frame = parts[-1]
        img_name = os.path.join('images', '_'.join([action, camera, frame]))

        annots = copy.deepcopy(self.annots[img_name_orig])
        img_path = os.path.join(self.dataset_path, img_name)

        cam_intrinsics = torch.from_numpy(annots['cam_intrinsics']).float().unsqueeze(0)
        cam_rot = torch.from_numpy(annots['cam_rot']).float().unsqueeze(0)
        cam_trans = torch.from_numpy(annots['cam_trans']).float().unsqueeze(0)

        betas = annots['betas']
        poses = torch.cat([annots['global_orient'].flatten(1), annots['body_pose'].flatten(1)], dim=1)
        transl = annots['transl']

        raw_data = {'img_path': img_path, 'ds': 'h36m', 'pnum': len(betas), 'betas': betas, 'poses': poses, 'transl': transl, 'cam_rot': cam_rot, 'cam_trans': cam_trans, 'cam_intrinsics': cam_intrinsics, '3d_valid': True, 'detect_all_people': True}

        return raw_data
