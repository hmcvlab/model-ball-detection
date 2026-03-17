"""DETR model and criterion classes."""

from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import ops

from . import backbone, transformer
from .position_encoding import build_position_encoding
from .segmentation import dice_loss, sigmoid_focal_loss
from .utils import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-arguments
# pylint: disable=too-few-public-methods


class DETR(nn.Module):
    """This is the DETR module that performs object detection."""

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """Initializes the model.

        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture.
                        See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is
                        the maximal number of objects DETR can detect in a
                        single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder
                    layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:

        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
        - samples.mask: a binary mask of shape [batch_size x H x W],
            containing 1 on padded pixels

        It returns a dict with the following elements:
        - "pred_logits": the classification logits (including no-object)
            for all queries. Shape= [batch_size x num_queries x (num_classes + 1)]
        - "pred_boxes": The normalized boxes coordinates for all queries,
            represented as (center_x, center_y, height, width). These values
            are normalized in [0, 1], relative to the size of each individual
            image (disregarding possible padding). See PostProcess for
            information on how to retrieve the unnormalized bounding box.
        - "aux_outputs": Optional, only returned when auxilary losses are
            activated. It is a list of dictionnaries containing the two above
            keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(
            self.input_proj(src), mask, self.query_embed.weight, pos[-1]
        )[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground
        truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth /
        prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """Create the criterion.

        Parameters:
            num_classes: number of object categories, omitting the special
                        no-object category
            matcher: module able to compute a matching between targets and
                    proposals
            weight_dict: dict containing as key the names of the losses and
                        as values their relative weight.
            eos_coef: relative classification weight applied to the no-object
                    category
            losses: list of all the losses to be applied. See get_loss for
                list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # pylint: disable=unused-argument
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL) targets dicts must contain the key "labels"
        containing a tensor of dim [nb_target_boxes]"""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][j] for t, (_, j) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}

        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # pylint: disable=unused-argument
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of
        predicted non-empty boxes This is not really a loss, it is intended for logging
        purposes only.

        It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and
        the GIoU loss targets dicts must contain the key "boxes" containing a tensor of
        dim [nb_target_boxes, 4] The target boxes are expected in format (center_x,
        center_y, w, h), normalized by the image size."""
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        box_iou = ops.box_iou(
            ops.box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
            ops.box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
        )
        loss_giou = 1 - torch.diag(box_iou)
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim
        [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.

        Parameters:
            outputs: dict of tensors, see the output specification of
                    the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                    The expected keys in each dict depends on the losses
                    applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes,
        # for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output
        # of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly
                        # to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """GPU-optimized conversion of DETR outputs to COCO / torchvision detection
    format."""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        Args:
            outputs (dict):
                Model outputs containing:
                    - "pred_logits": Tensor [B, Q, C+1]
                    - "pred_boxes":  Tensor [B, Q, 4] in cxcywh (relative coords)

            target_sizes (Tensor):
                Tensor [B, 2] with (height, width) of each image in the batch.

        Returns:
            List[Dict[str, Tensor]] (length B):
                Each dict contains:
                    - "scores": [Q]
                    - "labels": [Q]
                    - "boxes":  [Q, 4] in absolute xyxy format
        """

        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]

        probs = logits.softmax(-1)
        scores, labels = probs[..., :-1].max(-1)

        boxes = ops.box_convert(boxes, in_fmt="xcycwh", out_fmt="xyxy")

        scale = target_sizes.flip(1).repeat(1, 2)
        boxes.mul_(scale[:, None, :])

        results = []
        for i in range(boxes.shape[0]):
            results.append(
                {
                    "scores": scores[i],
                    "labels": labels[i],
                    "boxes": boxes[i],
                }
            )

        return results


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_backbone(
    hidden_dim, position_embedding, lr_backbone, masks, backbone, dilation
):
    """Build backbone model.

    Args:
        args: Arguments for building the backbone

    Returns:
        Backbone model
    """
    position_embedding = build_position_encoding(hidden_dim, position_embedding)
    train_backbone = lr_backbone > 0
    return_interm_layers = masks
    backbone = backbone.Backbone(
        backbone, train_backbone, return_interm_layers, dilation
    )
    model = backbone.Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def build(num_classes: int):
    """Build a DETR model with given args."""
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    params_backbone = backbone.Parameters()
    params_transformer = transformer.Parameters()

    backbone_obj = build_backbone(
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        lr_backbone=args.lr_backbone,
        masks=args.masks,
        backbone=args.backbone,
        dilation=args.dilation,
    )
    transformer_obj = transformer.Transformer(**asdict(params_transformer))

    model = DETR(
        backbone_obj,
        transformer_obj,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    matcher = HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
    )
    weight_dict = {"loss_ce": 1, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
    )
    criterion.to(device)
    postprocessors = {"bbox": PostProcess()}

    return model, criterion, postprocessors
