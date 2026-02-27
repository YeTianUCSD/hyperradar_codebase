import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_head_template import AnchorHeadTemplate
from ..hd.hd_core import HDCore


class AnchorHeadSingle(AnchorHeadTemplate):
    """
    PointPillars AnchorHeadSingle with optional HD (Hyperdimensional Computing) branch.

    This version supports two variants for the feature used by HD:
      - Option A (old): use BEV feature map (spatial_features_2d)
      - Option B (new): use "penultimate" feature inside the cls head (cls_feat)

    Key idea for Option B:
      - Replace 1-layer cls head (conv_cls) with 2-layer head:
          cls_feat = conv_cls_pre(spatial_features_2d)        # penultimate feature
          cls_logits = conv_cls_out(cls_feat)                 # final logits
      - Run HD encoding / logits on cls_feat instead of spatial_features_2d,
        so HD is closer to replacing the linear classifier.

    Config knobs (optional; safe defaults if missing):
      MODEL.DENSE_HEAD.CLS_FEAT_CHANNELS: int (default = input_channels)
      MODEL.DENSE_HEAD.CLS_PRE_KERNEL:    int (default = 3)
      MODEL.DENSE_HEAD.CLS_PRE_BN:        bool (default = True)
      MODEL.DENSE_HEAD.CLS_PRE_RELU:      bool (default = True)

      MODEL.DENSE_HEAD.HD.FEAT_SOURCE: "bev" | "cls" (default = "bev")
        - "bev": use spatial_features_2d for HD (old behavior)
        - "cls": use cls_feat for HD (new option B)
    """

    def __init__(
        self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
        predict_boxes_when_training=True, **kwargs
    ):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names,
            grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        # In OpenPCDet, num_anchors_per_location is a list per class; sum gives total A.
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        # -----------------------------
        # Box regression head (unchanged)
        # -----------------------------
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size, kernel_size=1
        )

        # -----------------------------
        # Direction head (unchanged)
        # -----------------------------
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None

        # -----------------------------
        # Classification head (NEW: 2-layer so we can take penultimate feature)
        # -----------------------------
        cls_feat_channels = int(self.model_cfg.get('CLS_FEAT_CHANNELS', input_channels))
        cls_pre_kernel = int(self.model_cfg.get('CLS_PRE_KERNEL', 3))
        cls_pre_bn = bool(self.model_cfg.get('CLS_PRE_BN', True))
        cls_pre_relu = bool(self.model_cfg.get('CLS_PRE_RELU', True))

        padding = cls_pre_kernel // 2
        layers = [
            nn.Conv2d(input_channels, cls_feat_channels, kernel_size=cls_pre_kernel, padding=padding, bias=not cls_pre_bn)
        ]
        if cls_pre_bn:
            layers.append(nn.BatchNorm2d(cls_feat_channels))
        if cls_pre_relu:
            layers.append(nn.ReLU(inplace=True))

        self.conv_cls_pre = nn.Sequential(*layers)

        # final classifier (same as original conv_cls)
        self.conv_cls_out = nn.Conv2d(
            cls_feat_channels, self.num_anchors_per_location * self.num_class, kernel_size=1
        )

        # Keep for convenience/logs
        self.cls_feat_channels = cls_feat_channels

        # -----------------------------
        # HD configs
        # -----------------------------
        hd_cfg = self.model_cfg.get('HD', None)

        self.hd_enabled = False
        self.hd_mode = "baseline"   # "baseline" | "hd_only" | "fused"
        self.hd_lambda = 0.5

        # If True, run assign_targets even in eval when GT exists (useful for building memory with GT)
        self.hd_assign_targets_in_eval = False

        # If True, store extra tensors in data_dict for online update/debug (can be large!)
        self.hd_export_for_online = False

        # If True, store origin/hd logits into forward_ret_dict for analysis
        self.hd_debug_save_logits = False
        # If True, allow gradients through HD path during training.
        self.hd_train_use_grad = False

        # Which feature map does HD use?
        #   - "bev": spatial_features_2d
        #   - "cls": cls_feat (penultimate)
        self.hd_feat_source = "bev"

        self.hd_core = None
        if hd_cfg is not None and bool(hd_cfg.get("ENABLED", False)):
            self.hd_enabled = True
            self.hd_mode = str(hd_cfg.get("MODE", "fused")).lower()
            self.hd_lambda = float(hd_cfg.get("LAMBDA", 0.5))
            self.hd_assign_targets_in_eval = bool(hd_cfg.get("ASSIGN_TARGETS_IN_EVAL", False))
            self.hd_export_for_online = bool(hd_cfg.get("EXPORT_FOR_ONLINE", False))
            self.hd_debug_save_logits = bool(hd_cfg.get("DEBUG_SAVE_LOGITS", False))
            self.hd_train_use_grad = bool(hd_cfg.get("TRAIN_USE_GRAD", False))

            # NEW: feature source for HD
            self.hd_feat_source = str(hd_cfg.get("FEAT_SOURCE", "bev")).lower()
            if self.hd_feat_source not in ("bev", "cls"):
                self.hd_feat_source = "bev"

            # IMPORTANT:
            # If HD uses cls_feat, feat_dim must match cls_feat_channels.
            # If HD uses bev, feat_dim matches input_channels.
            hd_feat_dim = cls_feat_channels if self.hd_feat_source == "cls" else input_channels

            self.hd_core = HDCore.from_cfg(
                hd_cfg=hd_cfg,
                feat_dim=hd_feat_dim,
                num_classes=self.num_class
            )

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        # cls bias init goes to final cls conv (conv_cls_out)
        nn.init.constant_(self.conv_cls_out.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
        # (optional) you can init conv_cls_pre conv weight, but defaults are fine

    @staticmethod
    def _as_float32(x: torch.Tensor) -> torch.Tensor:
        return x.float() if x.dtype != torch.float32 else x

    def forward(self, data_dict):
        """
        Forward flow:
          spatial_features_2d -> cls_head (pre + out) / box_head -> permute -> (optional) assign_targets -> decode boxes

        With HD enabled:
          - compute HD logits per CELL: [B*H*W, C_feat] -> [B*H*W, K]
          - repeat across anchors: [B,H,W,K] -> [B,H,W,A*K]
          - fuse with original cls logits (origin) by mode/lambda
        """
        spatial_features_2d = data_dict['spatial_features_2d']  # [B, C_in, H, W]
        B, C_in, H, W = spatial_features_2d.shape
        A = int(self.num_anchors_per_location)
        K = int(self.num_class)

        # -----------------------------
        # CLS head (2-layer)
        # -----------------------------
        cls_feat = self.conv_cls_pre(spatial_features_2d)     # [B, C_cls, H, W]
        data_dict["hd_cls_feat"] = cls_feat
        cls_preds_origin = self.conv_cls_out(cls_feat)        # [B, A*K, H, W]

        # -----------------------------
        # BOX head (unchanged)
        # -----------------------------
        box_preds = self.conv_box(spatial_features_2d)        # [B, A*code, H, W]

        # Permute to match OpenPCDet expected format
        cls_preds_origin = cls_preds_origin.permute(0, 2, 3, 1).contiguous()  # [B, H, W, A*K]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()                # [B, H, W, A*code]

        cls_preds_final = cls_preds_origin
        cls_preds_hd = None  # [B,H,W,A*K] if computed

        # -----------------------------
        # HD logits + fusion
        # -----------------------------
        if self.hd_enabled and (self.hd_core is not None):
            # choose feature source for HD
            if self.hd_feat_source == "cls":
                feat_map = cls_feat
                C_feat = feat_map.shape[1]
            else:
                feat_map = spatial_features_2d
                C_feat = C_in

            # Compute HD logits per anchor (anchor-aware): [B,H,W,A,C] -> [B,H,W,A,K]
            use_grad_for_hd = bool(self.training and self.hd_train_use_grad)
            with torch.set_grad_enabled(use_grad_for_hd):
                # [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
                cell_feat = feat_map.permute(0, 2, 3, 1).contiguous()
                feat_anchor = cell_feat.unsqueeze(3).expand(B, H, W, A, C_feat).reshape(-1, C_feat)
                anchor_ids = torch.arange(A, device=feat_anchor.device).view(1, 1, 1, A).expand(B, H, W, A).reshape(-1)

                # Inject anchor context before HD encoding (no-op when ANCHOR_ID_SCALE <= 0).
                feat_anchor = self.hd_core.inject_anchor_context(
                    feat_mid=feat_anchor,
                    anchor_ids=anchor_ids,
                    num_anchors=A
                )

                logits_anchor, _ = self.hd_core.compute_hd_logits(feat_anchor)  # [B*H*W*A, K]
                cls_preds_hd = logits_anchor.view(B, H, W, A, K).reshape(B, H, W, A * K).contiguous()

                # fuse
                cls_fused = self.hd_core.fuse_logits(
                    logits_origin=self._as_float32(cls_preds_origin),
                    logits_hd=self._as_float32(cls_preds_hd),
                    mode=self.hd_mode,
                    lam=self.hd_lambda
                )
                cls_preds_final = cls_fused  # keep float32 for safety

        # Save outputs used by later stages
        self.forward_ret_dict['cls_preds'] = cls_preds_final
        self.forward_ret_dict['box_preds'] = box_preds

        # Optional debug: store origin & hd logits for analysis
        if self.hd_enabled and self.hd_debug_save_logits:
            self.forward_ret_dict['cls_preds_origin'] = cls_preds_origin
            if cls_preds_hd is not None:
                self.forward_ret_dict['cls_preds_hd'] = cls_preds_hd

        # -----------------------------
        # Direction classifier (unchanged)
        # -----------------------------
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        # -----------------------------
        # Target assignment (GT labels)
        # -----------------------------
        need_assign = self.training
        if (not self.training) and self.hd_enabled and self.hd_assign_targets_in_eval:
            if ('gt_boxes' in data_dict) and (data_dict['gt_boxes'] is not None):
                need_assign = True

        if need_assign:
            targets_dict = self.assign_targets(gt_boxes=data_dict['gt_boxes'])
            self.forward_ret_dict.update(targets_dict)

            # Optional export for online update/debug:
            # WARNING: exporting tensors can be large.
            if self.hd_enabled and self.hd_export_for_online:
                # export BOTH sources so downstream scripts can choose consistently
                data_dict['hd_bev_feat'] = spatial_features_2d.detach()  # [B, C_in, H, W]
                data_dict['hd_cls_feat'] = cls_feat.detach()             # [B, C_cls, H, W]
                data_dict['hd_shape_bev'] = (int(B), int(C_in), int(H), int(W))
                data_dict['hd_shape_cls'] = (int(B), int(self.cls_feat_channels), int(H), int(W))
                data_dict['hd_num_anchors_per_loc'] = int(A)
                data_dict['hd_num_classes'] = int(K)
                data_dict['hd_feat_source'] = str(self.hd_feat_source)

        # -----------------------------
        # Decode predicted boxes (unchanged)
        # -----------------------------
        if (not self.training) or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds_final, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
