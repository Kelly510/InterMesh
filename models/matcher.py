# Modified from DAB-DETR (https://github.com/IDEA-Research/DAB-DETR)

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_conf: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_kpts: float = 10,
        j2ds_norm_scale: float = 518,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_conf = cost_conf
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_kpts = cost_kpts
        self.j2ds_norm_scale = j2ds_norm_scale
        assert cost_conf != 0 or cost_bbox != 0 or cost_giou != 0 or cost_kpts != 0, "all costs cant be 0"

        # self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        assert outputs['pred_confs'].shape[0] == len(targets)
        bs, num_queries, _ = outputs["pred_confs"].shape

        # We flatten to compute the cost matrices in a batch
        out_conf = outputs['pred_confs'].flatten(0, 1)  # [batch_size * num_queries, 1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_kpts = outputs['pred_j2ds'][..., :22, :].flatten(2).flatten(0, 1) / self.j2ds_norm_scale

        # Also concat the target labels and boxes
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_kpts = torch.cat([v['j2ds'][:, :22, :].flatten(1) for v in targets]) / self.j2ds_norm_scale
        tgt_kpts_mask = torch.cat([v['j2ds_mask'][:, :22, :].flatten(1) for v in targets])
        tgt_kpts_vis_cnt = tgt_kpts_mask.sum(-1)
        assert (torch.all(tgt_kpts_vis_cnt))

        # Compute the confidence cost.
        alpha = 0.25
        gamma = 2.0
        cost_conf = alpha * ((1 - out_conf)**gamma) * (-(out_conf + 1e-8).log())
        # cost_conf = -(out_conf+1e-8).log()

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # import ipdb; ipdb.set_trace()
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Compute the mean L1 cost between visible joints
        all_dist = torch.abs(out_kpts[:, None, :] - tgt_kpts[None, :, :])
        mean_dist = (all_dist * tgt_kpts_mask[None, :, :]).sum(-1) / tgt_kpts_vis_cnt[None, :]
        cost_kpts = mean_dist

        # Final cost matrix
        C = self.cost_conf * cost_conf + self.cost_kpts * cost_kpts + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_conf=args.set_cost_conf, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, cost_kpts=args.set_cost_kpts, j2ds_norm_scale=args.input_size)
