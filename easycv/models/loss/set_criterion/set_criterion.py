# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models.detection.utils import (accuracy, box_cxcywh_to_xyxy,
                                           box_iou, generalized_box_iou)
from easycv.models.loss.focal_loss import py_sigmoid_focal_loss
from easycv.models.utils import get_world_size, is_dist_avail_and_initialized

from typing import Optional, List
from torch import Tensor

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 losses,
                 eos_coef=None,
                 loss_class_type='ce'):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.loss_class_type = loss_class_type
        if self.loss_class_type == 'ce':
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = eos_coef
            self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device)
        target_classes[idx] = target_classes_o

        if self.loss_class_type == 'ce':
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight)
        elif self.loss_class_type == 'focal_loss':
            target_classes_onehot = torch.zeros([
                src_logits.shape[0], src_logits.shape[1],
                src_logits.shape[2] + 1
            ],
                                                dtype=src_logits.dtype,
                                                layout=src_logits.layout,
                                                device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]

            loss_ce = py_sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                alpha=0.25,
                gamma=2,
                reduction='none').mean(1).sum() / num_boxes
            loss_ce = loss_ce * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx],
                                                   target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets],
                                      device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_centerness(self, outputs, targets, indices, num_boxes):

        def ref2ltrb(ref, xyxy):
            lt = ref - xyxy[..., :2]
            rb = xyxy[..., 2:] - ref
            ltrb = torch.cat([lt, rb], dim=-1)
            return ltrb

        def compute_centerness_targets(box_targets):
            left_right = box_targets[:, [0, 2]]
            top_bottom = box_targets[:, [1, 3]]
            centerness = (left_right.min(-1)[0] / left_right.max(-1)[0]) * (
                top_bottom.min(-1)[0] / top_bottom.max(-1)[0])
            return torch.sqrt(centerness)

        assert 'pred_centers' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_centers = outputs['pred_centers'][idx]  # logits
        src_centers = src_centers.squeeze(1)
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        assert 'refpts' in outputs
        src_refpts = outputs['refpts'][idx]  # sigmoided
        assert src_refpts.shape[-1] == 2

        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        target_boxes_ltrb = ref2ltrb(src_refpts, target_boxes_xyxy)
        is_in_box = torch.sum(target_boxes_ltrb >= 0, dim=-1) == 4

        src_centers = src_centers[is_in_box]
        target_boxes_ltrb = target_boxes_ltrb[is_in_box]

        target_boxes_ltrb = target_boxes_ltrb.detach()

        losses = {}
        if len(target_boxes_ltrb) == 0:
            losses['loss_center'] = src_centers.sum(
            ) * 0  # prevent unused parameters
        else:
            target_centers = compute_centerness_targets(target_boxes_ltrb)
            loss_center = F.binary_cross_entropy_with_logits(
                src_centers, target_centers, reduction='none')
            losses['loss_center'] = loss_center.sum() / num_boxes

        return losses

    def loss_iouaware(self, outputs, targets, indices, num_boxes):
        assert 'pred_ious' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_ious = outputs['pred_ious'][idx]  # logits
        src_ious = src_ious.squeeze(1)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        iou = torch.diag(
            box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes))[0])

        losses = {}
        loss_iouaware = F.binary_cross_entropy_with_logits(
            src_ious, iou, reduction='none')
        losses['loss_iouaware'] = loss_iouaware.sum() / num_boxes
        return losses

    def loss_tokens(self, outputs, targets, num_boxes):
        enc_token_class_unflat = outputs['pred_tokens']

        def _max_by_axis(the_list):
            # type: (List[List[int]]) -> List[int]
            maxes = the_list[0]
            for sublist in the_list[1:]:
                for index, item in enumerate(sublist):
                    maxes[index] = max(maxes[index], item)
            return maxes

        def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
            # TODO make this more general
            if tensor_list[0].ndim == 3:
                # TODO make it support different-sized images
                max_size = _max_by_axis([list(img.shape) for img in tensor_list])
                # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
                batch_shape = [len(tensor_list)] + max_size
                b, c, h, w = batch_shape
                dtype = tensor_list[0].dtype
                device = tensor_list[0].device
                tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
                mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
                for img, pad_img, m in zip(tensor_list, tensor, mask):
                    pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                    m[: img.shape[1], :img.shape[2]] = False
            else:
                raise ValueError('not supported')
            return NestedTensor(tensor, mask)

        def dice_loss(inputs, targets, num_boxes):
            """
            Compute the DICE loss, similar to generalized IOU for masks
            Args:
                inputs: A float tensor of arbitrary shape.
                        The predictions for each example.
                targets: A float tensor with the same shape as inputs. Stores the binary
                        classification label for each element in inputs
                        (0 for the negative class and 1 for the positive class).
            """
            inputs = inputs.sigmoid()
            inputs = inputs.flatten(1)
            targets = targets.flatten(1)
            numerator = 2 * (inputs * targets).sum(1)
            denominator = inputs.sum(-1) + targets.sum(-1)
            loss = 1 - (numerator + 1) / (denominator + 1)
            return loss.sum() / num_boxes

        target_masks, valid = nested_tensor_from_tensor_list([t["masks"].to_tensor(dtype=torch.bool, device=enc_token_class_unflat[0].device) for t in targets]).decompose()

        bs, n, h, w = target_masks.shape
        mask = torch.zeros((bs, h, w), dtype=torch.bool, device=target_masks.device)
        for j in range(n):
            target_masks[:, j] &= target_masks[:, j] ^ mask
            mask |= target_masks[:, j]
        target_classes_pad = torch.stack([F.pad(t['labels'], (0, n - len(t['labels']))) for t in targets])
        final_mask = torch.sum(target_masks * target_classes_pad[:, :, None, None], dim=1)  # (bs, h, w)
        final_mask_onehot = torch.zeros((bs, h, w, self.num_classes), dtype=torch.float32, device=target_masks.device)
        final_mask_onehot.scatter_(-1, final_mask.unsqueeze(-1), 1)  # (bs, h, w, 80)
        final_mask_onehot[..., 0] = 1 - final_mask_onehot[..., 0]  # change index 0 from background to foreground

        loss_token_focal = 0
        loss_token_dice = 0
        for i, enc_token_class in enumerate(enc_token_class_unflat):
            _, h, w, _ = enc_token_class.shape

            final_mask_soft = F.adaptive_avg_pool2d(final_mask_onehot.permute(0, 3, 1, 2), (h,w)).permute(0, 2, 3, 1)

            enc_token_class = enc_token_class.flatten(1, 2)
            final_mask_soft = final_mask_soft.flatten(1, 2)
            loss_token_focal += py_sigmoid_focal_loss(
                                    enc_token_class,
                                    final_mask_soft,
                                    alpha=0.25,
                                    gamma=2,
                                    reduction='none').mean(1).sum() / num_boxes
            loss_token_dice += dice_loss(enc_token_class, final_mask_soft, num_boxes)

        losses = {
            'loss_token_focal': loss_token_focal,
            'loss_token_dice': loss_token_dice,
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'centerness': self.loss_centerness,
            'iouaware': self.loss_iouaware,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, num_boxes=None, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """

        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != 'aux_outputs'
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            indices0_copy = indices
            indices_list = []

        if num_boxes is None:
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t['labels']) for t in targets)
            num_boxes = torch.as_tensor([num_boxes],
                                        dtype=torch.float,
                                        device=next(iter(
                                            outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {
                k: v * (self.weight_dict[k] if k in self.weight_dict else 1.0)
                for k, v in l_dict.items()
            }
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           num_boxes, **kwargs)
                    l_dict = {
                        k + f'_{i}': v *
                        (self.weight_dict[k] if k in self.weight_dict else 1.0)
                        for k, v in l_dict.items()
                    }
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices,
                                       num_boxes, **kwargs)
                l_dict = {
                    k + '_interm':
                    v * (self.weight_dict[k] if k in self.weight_dict else 1.0)
                    for k, v in l_dict.items()
                }
                losses.update(l_dict)
            if 'pred_tokens' in interm_outputs and interm_outputs['pred_tokens'] is not None:
                l_dict = self.loss_tokens(interm_outputs, targets, num_boxes)
                l_dict = {
                    k + '_interm':
                    v * (self.weight_dict[k] if k in self.weight_dict else 1.0)
                    for k, v in l_dict.items()
                }
                losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


class CDNCriterion(SetCriterion):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 losses,
                 eos_coef=None,
                 loss_class_type='ce'):
        super().__init__(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            eos_coef=eos_coef,
            loss_class_type=loss_class_type)

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups

    def forward(self, outputs, targets, aux_num, num_boxes):
        # Compute the average number of target boxes accross all nodes, for normalization purposes

        dn_meta = outputs['dn_meta']
        losses = {}
        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(
                dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.range(0,
                                    len(targets[i]['labels']) -
                                    1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) *
                                  single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(
                    self.get_loss(loss, output_known_lbs_bboxes, targets,
                                  dn_pos_idx, num_boxes * scalar, **kwargs))

            l_dict = {
                k + '_dn':
                v * (self.weight_dict[k] if k in self.weight_dict else 1.0)
                for k, v in l_dict.items()
            }
            losses.update(l_dict)
        else:
            l_dict = dict()
            if 'labels' in self.losses:
                l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            if 'boxes' in self.losses:
                l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            if 'centerness' in self.losses:
                l_dict['loss_center_dn'] = torch.as_tensor(0.).to('cuda')
            if 'iouaware' in self.losses:
                l_dict['loss_iouaware_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        for i in range(aux_num):
            if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][i]
                l_dict = {}
                for loss in self.losses:
                    kwargs = {}
                    if 'labels' in loss:
                        kwargs = {'log': False}

                    l_dict.update(
                        self.get_loss(loss, aux_outputs_known, targets,
                                      dn_pos_idx, num_boxes * scalar,
                                      **kwargs))

                l_dict = {
                    k + f'_dn_{i}':
                    v * (self.weight_dict[k] if k in self.weight_dict else 1.0)
                    for k, v in l_dict.items()
                }
                losses.update(l_dict)
            else:
                l_dict = dict()
                if 'labels' in self.losses:
                    l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
                if 'boxes' in self.losses:
                    l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
                if 'centerness' in self.losses:
                    l_dict['loss_center_dn'] = torch.as_tensor(0.).to(
                        'cuda')
                if 'iouaware' in self.losses:
                    l_dict['loss_iouaware_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict = {
                    k + f'_{i}':
                    v * (self.weight_dict[k] if k in self.weight_dict else 1.0)
                    for k, v in l_dict.items()
                }
                losses.update(l_dict)
        return losses


class DNCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, weight_dict):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict

    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict[
            'output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice']

        known_indice = mask_dict['known_indice']

        batch_idx = mask_dict['batch_idx']
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(
                1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(
                1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt

    def tgt_loss_boxes(
        self,
        src_boxes,
        tgt_boxes,
        num_tgt,
    ):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if len(tgt_boxes) == 0:
            return {
                'loss_bbox': torch.as_tensor(0.).to('cuda'),
                'loss_giou': torch.as_tensor(0.).to('cuda'),
            }

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_tgt

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_tgt
        return losses

    def tgt_loss_labels(self,
                        src_logits_,
                        tgt_labels_,
                        num_tgt,
                        focal_alpha,
                        log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if len(tgt_labels_) == 0:
            return {
                'loss_ce': torch.as_tensor(0.).to('cuda'),
                'class_error': torch.as_tensor(0.).to('cuda'),
            }

        src_logits, tgt_labels = src_logits_.unsqueeze(
            0), tgt_labels_.unsqueeze(0)

        target_classes_onehot = torch.zeros([
            src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1
        ],
                                            dtype=src_logits.dtype,
                                            layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = py_sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            alpha=focal_alpha,
            gamma=2,
            reduction='none').mean(1).sum() / num_tgt * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits_, tgt_labels_)[0]
        return losses

    def forward(self, mask_dict, aux_num):
        """
        compute dn loss in criterion
        Args:
            mask_dict: a dict for dn information
            training: training or inference flag
            aux_num: aux loss number
        """
        losses = {}
        if self.training and 'output_known_lbs_bboxes' in mask_dict:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(
                mask_dict)
            l_dict = self.tgt_loss_labels(output_known_class[-1], known_labels,
                                          num_tgt, 0.25)
            l_dict = {
                k + '_dn':
                v * (self.weight_dict[k] if k in self.weight_dict else 1.0)
                for k, v in l_dict.items()
            }
            losses.update(l_dict)
            l_dict = self.tgt_loss_boxes(output_known_coord[-1], known_bboxs,
                                         num_tgt)
            l_dict = {
                k + '_dn':
                v * (self.weight_dict[k] if k in self.weight_dict else 1.0)
                for k, v in l_dict.items()
            }
            losses.update(l_dict)
        else:
            losses['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            losses['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            losses['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')

        if aux_num:
            for i in range(aux_num):
                # dn aux loss
                if self.training and 'output_known_lbs_bboxes' in mask_dict:
                    l_dict = self.tgt_loss_labels(output_known_class[i],
                                                  known_labels, num_tgt, 0.25)
                    l_dict = {
                        k + f'_dn_{i}': v *
                        (self.weight_dict[k] if k in self.weight_dict else 1.0)
                        for k, v in l_dict.items()
                    }
                    losses.update(l_dict)
                    l_dict = self.tgt_loss_boxes(output_known_coord[i],
                                                 known_bboxs, num_tgt)
                    l_dict = {
                        k + f'_dn_{i}': v *
                        (self.weight_dict[k] if k in self.weight_dict else 1.0)
                        for k, v in l_dict.items()
                    }
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {
                        k + f'_{i}': v *
                        (self.weight_dict[k] if k in self.weight_dict else 1.0)
                        for k, v in l_dict.items()
                    }
                    losses.update(l_dict)
        return losses
