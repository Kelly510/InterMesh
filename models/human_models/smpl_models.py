import torch
from torch import nn
import smplx
import numpy as np
import pickle
import os.path as osp
from configs.paths import smpl_model_path


class SMPL_Layer(nn.Module):

    def __init__(self, model_path, with_genders=True, **kwargs):
        """
        Extension of the SMPL Layer with gendered inputs.
        """
        super().__init__()
        smpl_kwargs = {'create_global_orient': False, 'create_body_pose': False, 'create_betas': False, 'create_transl': False}
        smpl_kwargs.update(kwargs)
        self.with_genders = with_genders
        if self.with_genders:
            self.layer_n = smplx.create(model_path, 'smpl', gender='neutral', **smpl_kwargs)
            self.layer_m = smplx.create(model_path, 'smpl', gender='male', **smpl_kwargs)
            self.layer_f = smplx.create(model_path, 'smpl', gender='female', **smpl_kwargs)
            self.layers = {'neutral': self.layer_n, 'male': self.layer_m, 'female': self.layer_f}
        else:
            self.layer_n = smplx.create(model_path, 'smpl', gender='neutral', **smpl_kwargs)
            self.layers = {'neutral': self.layer_n}

        self.vertex_num = 6890
        self.faces = self.layer_n.faces

        self.body_vertex_idx = np.load(osp.join(model_path, 'smpl', 'body_verts_smpl.npy'))
        self.smpl2h36m_regressor = np.load(osp.join(model_path, 'smpl', 'J_regressor_h36m_correct.npy'))

    def forward_single_gender(self, poses, betas, gender='neutral'):
        bs = poses.shape[0]
        if poses.ndim == 2:
            poses = poses.view(bs, -1, 3)

        assert poses.shape[1] == 24
        pose_params = {'global_orient': poses[:, :1, :], 'body_pose': poses[:, 1:, :]}

        smpl_output = self.layers[gender](betas=betas, **pose_params)
        return smpl_output.vertices, smpl_output.joints

    def forward(self, poses, betas, genders=None):
        bs = poses.shape[0]
        assert poses.shape[0] == betas.shape[0]
        if genders is None:
            return self.forward_single_gender(poses, betas)
        else:
            assert len(genders) == bs
            assert set(genders) <= {'male', 'female'}
            assert self.with_genders

            male_idx = [i for i, gender in enumerate(genders) if gender == 'male']
            if len(male_idx) == bs:
                return self.forward_single_gender(poses, betas, gender='male')
            elif len(male_idx) == 0:
                return self.forward_single_gender(poses, betas, gender='female')
            else:
                vertices, joints = self.forward_single_gender(poses, betas, gender='female')
                vertices[male_idx], joints[male_idx] =\
                         self.forward_single_gender(poses[male_idx], betas[male_idx], gender='male')
                return vertices, joints


smpl_gendered = SMPL_Layer(smpl_model_path, with_genders=True)
