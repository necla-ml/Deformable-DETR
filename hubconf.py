from argparse import Namespace
from models import build_model as build

dependencies = ["torch", "torchvision"]


def build_deformable_detr(panoptic=False, num_classes=91, **kwargs):
    args = Namespace()
    args.dataset_file = 'coco_panoptic' if panoptic else 'coco'
    args.device = kwargs.get('device', 'cuda')
    args.num_classes = num_classes
    args.num_feature_levels = kwargs.get('feature_levels', 4)
    args.aux_loss = kwargs.get('aux_loss', True)
    args.with_box_refine = kwargs.get('with_box_refine', False)
    args.masks = kwargs.get('masks', False)
    args.mask_loss_coef = kwargs.get('mask_loss_coef', 1.0)
    args.dice_loss_coef = kwargs.get('dice_loss_coef', 1.0)
    args.cls_loss_coef = kwargs.get('cls_loss_coef', 2.0)
    args.bbox_loss_coef = kwargs.get('bbox_loss_coef', 5.0)
    args.giou_loss_coef = kwargs.get('giou_loss_coef', 2.0)
    args.focal_alpha = kwargs.get('focal_alpha', 0.25)
    args.frozen_weights = kwargs.get('frozen_weights', None)

    # backbone
    args.backbone = kwargs.get('backbone', 'resnet50')
    args.lr_backbone = kwargs.get('lr_backbone', 2e-5)
    args.dilation = kwargs.get('dilation', False)

    # positional encoding
    args.position_embedding = kwargs.get('position_embedding', 'sine') # learned
    args.hidden_dim = kwargs.get('hidden_dim', 256)
    
    # transformer
    args.nheads = kwargs.get('nheads', 8)
    args.dim_feedforward = kwargs.get('dim_feedforward', 1024)
    args.enc_layers = kwargs.get('enc_layers', 6)
    args.dec_layers = kwargs.get('dec_layers', 6)
    args.dropout = kwargs.get('dropout', 0.1)
    args.dec_n_points = kwargs.get('dec_n_points', 4)
    args.enc_n_points = kwargs.get('enc_n_points', 4)
    args.num_queries = kwargs.get('num_queries', 300)
    args.two_stage = kwargs.get('two_stage', False)

    # loss
    args.set_cost_class = kwargs.get('set_cost_class', 2) 
    args.set_cost_bbox = kwargs.get('set_cost_bbox', 5) 
    args.set_cost_giou = kwargs.get('set_cost_giou', 2) 


    model, criterion, postprocessors = build(args)
    model.to(args.device)

    return_postprocessor = kwargs.get('return_postprocessors', False)
    if return_postprocessor:
        return model, postprocessors
    return model


def deformable_detr_r50(num_classes=91, return_postprocessor=False):
    if return_postprocessor:
        model, postprocessors = build_deformable_detr(backbone='resnet50', num_classes=num_classes, return_postprocessor=return_postprocessor)
        return model, postprocessors
    else:
        model = build_deformable_detr(backbone='resnet50', num_classes=num_classes, return_postprocessor=return_postprocessor)
        return model
