# Modified from DAB-DETR (https://github.com/IDEA-Research/DAB-DETR)
import os
import torch
from torch import nn
import torch.nn.functional as F
from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list, accuracy, get_world_size, interpolate, is_dist_avail_and_initialized, inverse_sigmoid)


def focal_loss(inputs, targets, detect_all_mask=None, alpha: float = 0.25, gamma: float = 2, lvl: str = "image"):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if detect_all_mask is not None:
        if lvl == 'image':
            valid_mask = detect_all_mask != 0
        elif lvl == 'instance':
            valid_mask = torch.logical_or(targets != 0, detect_all_mask != 0)
        else:
            raise ValueError
        return (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    else:
        return loss.mean()


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, losses=['confs', 'boxes', 'poses', 'betas', 'j3ds', 'j2ds', 'depths', 'kid_offsets'], focal_alpha=0.25, focal_gamma=2.0, j2ds_norm_scale=518, detect_all_mask_lvl='image'):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.losses = losses
        if 'boxes' in losses and 'giou' not in weight_dict:
            weight_dict.update({'giou': weight_dict['boxes']})
        self.weight_dict = weight_dict

        self.betas_weight = torch.tensor([2.56, 1.28, 0.64, 0.64, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32]).unsqueeze(0).float()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.j2ds_norm_scale = j2ds_norm_scale
        self.device = None
        self.detect_all_mask_lvl = detect_all_mask_lvl

    def loss_boxes(self, loss, outputs, targets, indices, num_instances, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        assert loss == 'boxes'
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device=self.device) * (loss in t) for t, (_, i) in zip(targets, indices)]))[0]

        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_' + loss][idx][valid_idx]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)
        assert src.shape == target.shape

        src_boxes = src
        target_boxes = target

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['boxes'] = loss_bbox.sum() / num_instances

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['giou'] = loss_giou.sum() / num_instances

        # # calculate the x,y and h,w loss
        # with torch.no_grad():
        #     losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #     losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    # For computing ['boxes', 'poses', 'betas', 'j3ds', 'j2ds'] losses
    def loss_L1(self, loss, outputs, targets, indices, num_instances, **kwargs):
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device=self.device) * (loss in t) for t, (_, i) in zip(targets, indices)]))[0]

        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_' + loss][idx][valid_idx]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)
        assert src.shape == target.shape

        losses = {}
        loss_mask = None

        if loss == 'j3ds':
            # Root aligned
            src = src - src[..., [0], :].clone()
            target = target - target[..., [0], :].clone()
            # Use 54 smpl joints
            src = src[:, :54, :]
            target = target[:, :54, :]
        elif loss == 'j2ds':
            src = src / self.j2ds_norm_scale
            target = target / self.j2ds_norm_scale
            # Need to exclude invalid kpts in 2d datasets
            loss_mask = torch.cat([t['j2ds_mask'][i] for t, (_, i) in zip(targets, indices) if 'j2ds' in t], dim=0)
            # Use 54 smpl joints
            src = src[:, :54, :]
            target = target[:, :54, :]
            loss_mask = loss_mask[:, :54, :]

        valid_loss = torch.abs(src - target)

        # if loss == 'j2ds':
        #     print(src.shape)
        #     print(target.shape)
        #     print(num_instances)
        #     exit(0)

        if loss_mask is not None:
            valid_loss = valid_loss * loss_mask
        if loss == 'betas':
            valid_loss = valid_loss * self.betas_weight.to(src.device)

        losses[loss] = valid_loss.flatten(1).mean(-1).sum() / num_instances

        return losses

    def loss_scale_map(self, loss, outputs, targets, indices, num_instances, **kwargs):
        assert loss == 'scale_map'

        pred_map = outputs['enc_outputs']['scale_map']
        tgt_map = torch.cat([t['scale_map'] for t in targets], dim=0)
        assert pred_map.shape == tgt_map.shape

        labels = tgt_map[:, 0]
        pred_scales = pred_map[:, 1]
        tgt_scales = tgt_map[:, 1]

        detect_all_mask = labels.bool()
        cur = 0
        lens = [len(t['scale_map']) for t in targets]
        for i, tgt in enumerate(targets):
            if tgt['detect_all_people']:
                detect_all_mask[cur:cur + lens[i]] = True
            cur += lens[i]

        losses = {}
        losses['map_confs'] = focal_loss(pred_map[:, 0], labels, detect_all_mask, lvl=self.detect_all_mask_lvl) / 1.
        losses['map_scales'] = torch.abs((pred_scales - tgt_scales)[torch.where(labels)[0]]).sum() / num_instances

        return losses

    def loss_confs(self, loss, outputs, targets, indices, num_instances, is_dn=False, **kwargs):
        assert loss == 'confs'
        idx = self._get_src_permutation_idx(indices)
        pred_confs = outputs['pred_' + loss]

        with torch.no_grad():
            labels = torch.zeros_like(pred_confs)
            labels[idx] = 1
            detect_all_mask = torch.zeros_like(pred_confs, dtype=bool)
            detect_all_mask[idx] = True
            valid_batch_idx = torch.where(torch.tensor([t['detect_all_people'] for t in targets]))[0]
            detect_all_mask[valid_batch_idx] = True

        losses = {}
        if is_dn:
            losses[loss] = focal_loss(pred_confs, labels) / num_instances
        else:
            losses[loss] = focal_loss(pred_confs, labels, detect_all_mask, lvl=self.detect_all_mask_lvl) / num_instances

        return losses

    def loss_absolute_depths(self, loss, outputs, targets, indices, num_instances, **kwargs):
        assert loss == 'depths'
        losses = {}
        idx = self._get_src_permutation_idx(indices)
        valid_idx = torch.where(torch.cat([torch.ones(len(i), dtype=bool, device=self.device) * (loss in t) for t, (_, i) in zip(targets, indices)]))[0]

        if len(valid_idx) == 0:
            return {loss: torch.tensor(0.).to(self.device)}

        src = outputs['pred_' + loss][idx][valid_idx][..., [1]]  # [d d/f]
        target = torch.cat([t[loss][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)[..., [0]]
        target_focals = torch.cat([t['focals'][i] for t, (_, i) in zip(targets, indices) if loss in t], dim=0)

        # print(src.shape, target.shape, target_focals.shape)

        src = target_focals * src

        assert src.shape == target.shape

        valid_loss = torch.abs(1. / (src + 1e-8) - 1. / (target + 1e-8))
        losses[loss] = valid_loss.flatten(1).mean(-1).sum() / num_instances
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            'confs': self.loss_confs,
            'boxes': self.loss_boxes,
            'poses': self.loss_L1,
            'betas': self.loss_L1,
            'j3ds': self.loss_L1,
            'j2ds': self.loss_L1,
            'depths': self.loss_absolute_depths,
            'scale_map': self.loss_scale_map,
        }
        # assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](loss, outputs, targets, indices, num_instances, **kwargs)

    def get_valid_instances(self, targets):
        # Compute the average number of GTs accross all nodes, for normalization purposes
        num_valid_instances = {}
        for loss in self.losses:
            num_instances = 0
            if loss != 'scale_map':
                for t in targets:
                    num_instances += t['pnum'] if loss in t else 0
                num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=self.device)
            else:
                for t in targets:
                    num_instances += t['scale_map'][..., 0].sum().item()
                num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=self.device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_instances)
            num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()
            num_valid_instances[loss] = num_instances
        num_valid_instances['confs'] = 1.
        return num_valid_instances

    def prep_for_dn(self, dn_meta):
        output_known = dn_meta['output_known']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known, single_pad, num_dn_groups

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # remove invalid information in targets
        for t in targets:
            if not t['3d_valid']:
                for key in ['betas', 'poses', 'j3ds', 'depths', 'focals']:
                    if key in t:
                        del t[key]

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'sat'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        self.device = outputs['pred_poses'].device
        num_valid_instances = self.get_valid_instances(targets)

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        if 'dn_meta' in outputs:
            dn_meta = outputs['dn_meta']
            output_known, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                assert len(targets[i]['boxes']) > 0
                # t = torch.range(0, len(targets[i]['labels']) - 1).long().to(self.device)
                t = torch.arange(0, len(targets[i]['labels'])).long().to(self.device)
                t = t.unsqueeze(0).repeat(scalar, 1)
                tgt_idx = t.flatten()
                output_idx = (torch.tensor(range(scalar)) * single_pad).long().to(self.device).unsqueeze(1) + t
                output_idx = output_idx.flatten()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            l_dict = {}
            for loss in self.losses:
                if loss == 'scale_map':
                    continue
                l_dict.update(self.get_loss(loss, output_known, targets, dn_pos_idx, num_valid_instances[loss] * scalar, is_dn=True))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_valid_instances[loss]))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'scale_map':
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_valid_instances[loss])
                    l_dict = {f'{k}.{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if 'dn_meta' in outputs:
                    if loss == 'scale_map':
                        continue
                    aux_outputs_known = output_known['aux_outputs'][i]
                    l_dict = {}
                    for loss in self.losses:
                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_valid_instances[loss] * scalar, is_dn=True))
                    l_dict = {k + f'_dn.{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # if 'scale_map' in outputs:
        #     enc_outputs = outputs['enc_outputs']
        #     indices = self.matcher.forward_enc(enc_outputs, targets)
        #     for loss in ['confs_enc', 'boxes_enc']:
        #         l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_valid_instances[loss.replace('_enc','')])
        #         l_dict = {k + f'_enc': v for k, v in l_dict.items()}
        #         losses.update(l_dict)

        return losses
