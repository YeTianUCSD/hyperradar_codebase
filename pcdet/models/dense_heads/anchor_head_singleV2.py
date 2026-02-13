import numpy as np
import torch
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
from ..hd.hd_core import HDCore


class AnchorHeadSingle(AnchorHeadTemplate):
    """
    PointPillars AnchorHeadSingle with optional HD (Hyperdimensional Computing) branch.

    Path-1 (fast baseline / hd-only / fused ablation):
      feat_mid := spatial_features_2d (cell-level BEV feature)
      - Compute HD logits per CELL only: [B*H*W, C] -> [B*H*W, K]
      - Repeat to anchors at that cell to match [B, H, W, A*K]
      - Fuse with original conv_cls logits

    Notes:
      - We DO NOT expand cell features to anchor-level for HD encoding to avoid OOM.
      - assign_targets outputs keys used by AnchorHeadTemplate losses:
          'box_cls_labels', 'box_reg_targets', etc.
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
        # Original PointPillars heads
        # -----------------------------
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None

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

        self.hd_core = None
        if hd_cfg is not None and bool(hd_cfg.get("ENABLED", False)):
            self.hd_enabled = True
            self.hd_mode = str(hd_cfg.get("MODE", "fused")).lower()
            self.hd_lambda = float(hd_cfg.get("LAMBDA", 0.5))
            self.hd_assign_targets_in_eval = bool(hd_cfg.get("ASSIGN_TARGETS_IN_EVAL", False))
            self.hd_export_for_online = bool(hd_cfg.get("EXPORT_FOR_ONLINE", False))
            self.hd_debug_save_logits = bool(hd_cfg.get("DEBUG_SAVE_LOGITS", False))

            # Path-1: feat_mid = spatial_features_2d (cell-level feature, dim=input_channels)
            self.hd_core = HDCore.from_cfg(
                hd_cfg=hd_cfg,
                feat_dim=input_channels,
                num_classes=self.num_class
            )

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        """
        Forward flow:
          spatial_features_2d -> conv_cls/conv_box -> permute -> (optional) assign_targets -> (optional) decode boxes

        With HD enabled (Path-1 optimized):
          - cell_feat = spatial_features_2d per BEV cell
          - compute HD logits once per cell: [B*H*W, C] -> [B*H*W, K]
          - repeat across anchors: [B,H,W,K] -> [B,H,W,A,K] -> [B,H,W,A*K]
          - fuse with origin logits according to HD.MODE & HD.LAMBDA
        """
        spatial_features_2d = data_dict['spatial_features_2d']  # [B, C_in, H, W]
        B, C_in, H, W = spatial_features_2d.shape
        A = int(self.num_anchors_per_location)
        K = int(self.num_class)

        # -----------------------------
        # Original predictions
        # -----------------------------
        cls_preds_origin = self.conv_cls(spatial_features_2d)  # [B, A*K, H, W]
        box_preds = self.conv_box(spatial_features_2d)         # [B, A*code, H, W]

        # Permute to match OpenPCDet expected format
        cls_preds_origin = cls_preds_origin.permute(0, 2, 3, 1).contiguous()  # [B, H, W, A*K]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()                # [B, H, W, A*code]

        cls_preds_final = cls_preds_origin
        cls_preds_hd = None  # [B,H,W,A*K] if computed

        # -----------------------------
        # HD logits + fusion (Path-1 optimized)
        # -----------------------------
        if self.hd_enabled and (self.hd_core is not None):
            # We compute HV/logits per CELL (B*H*W) once, then repeat to anchors.
            # This avoids expanding to anchor-level and prevents OOM.
            with torch.no_grad():
                # [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
                cell_feat = spatial_features_2d.permute(0, 2, 3, 1).contiguous()
                feat_cell = cell_feat.view(-1, C_in)  # [B*H*W, C]

                # HD logits per cell: [B*H*W, K]
                logits_cell, _hv_cell = self.hd_core.compute_hd_logits(feat_cell)

                # Reshape to [B, H, W, K]
                logits_cell = logits_cell.view(B, H, W, K)

                # Repeat across anchors: [B,H,W,K] -> [B,H,W,A,K] -> [B,H,W,A*K]
                cls_preds_hd = logits_cell.unsqueeze(3).expand(B, H, W, A, K).reshape(B, H, W, A * K).contiguous()

                # dtype safety: make sure both are float32 before fusing (avoid AMP edge cases)
                cls_origin_f = cls_preds_origin.float()
                cls_hd_f = cls_preds_hd.float()

                cls_fused = self.hd_core.fuse_logits(
                    logits_origin=cls_origin_f,
                    logits_hd=cls_hd_f,
                    mode=self.hd_mode,
                    lam=self.hd_lambda
                )

                # Keep output dtype consistent with OpenPCDet expectation (float32 is safest)
                cls_preds_final = cls_fused

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
        # Default: only in training.
        # Optional (HD): run in eval as well IF GT exists and enabled.
        need_assign = self.training
        if (not self.training) and self.hd_enabled and self.hd_assign_targets_in_eval:
            if ('gt_boxes' in data_dict) and (data_dict['gt_boxes'] is not None):
                need_assign = True

        if need_assign:
            targets_dict = self.assign_targets(gt_boxes=data_dict['gt_boxes'])
            self.forward_ret_dict.update(targets_dict)

            # For later online/buffer scripts:
            # AnchorHeadTemplate losses and typical pipelines use:
            #   - 'box_cls_labels' (shape [B, num_anchors])
            #   - 'box_reg_targets'
            # So if you need labels, use forward_ret_dict['box_cls_labels'].

            # Optional export for online update/debug:
            # WARNING: exporting tensors can be large. Here we export only cell-level features
            # and shape meta so the runner can reconstruct per-cell or per-anchor consistently.
            if self.hd_enabled and self.hd_export_for_online:
                data_dict['hd_cell_feat'] = spatial_features_2d.detach()  # [B, C, H, W]
                data_dict['hd_shape'] = (int(B), int(C_in), int(H), int(W))
                data_dict['hd_num_anchors_per_loc'] = int(A)
                data_dict['hd_num_classes'] = int(K)

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
