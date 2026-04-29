#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model loading and single-image inference wrapper.

Handles checkpoint loading, task configuration, and provides a simple
``infer_task()`` interface used by the rest of the demo pipeline.
"""

from __future__ import annotations

from typing import Any

import torch
from easydict import EasyDict as edict

# Ensure Moeface_project is on sys.path before importing core.*
from . import config  # noqa: F401 — side-effect: adds MOEFACE_PROJECT_DIR to sys.path

from core.model.backbone import backbone_entry
from core.model.heads import heads_holder_entry
from core.model.loss import LossesHolder
from core.model.model_entry import AIOEntry


# ============================================================================
# Task config builder
# ============================================================================

def build_task_cfgs_for_aio(cfg: edict) -> list[edict]:
    """Build the ordered list of task config dicts expected by AIOEntry."""
    tasks: list[edict] = []

    ordered = [
        ("recog",    cfg.tasks.recog.task_name),
        ("age",      cfg.tasks.age.task_name),
        ("biattr",   cfg.tasks.attribute.task_name),
        ("affect",   cfg.tasks.expression.task_name),
        ("parsing",  cfg.tasks.parsing.task_name),
        ("align",    cfg.tasks.align.task_name),
    ]

    for task_type, task_name in ordered:
        loss_cfg = _build_loss_cfg(task_type)
        tasks.append(edict({
            "name": task_name,
            "loss_weight": 1.0,
            "sampler": edict({"batch_size": 1}),
            "loss": loss_cfg,
        }))

    return tasks


def _build_loss_cfg(task_type: str) -> edict:
    """Return the loss configuration for a given task type."""
    mapping = {
        "recog": ("MarginCosineProductLoss", {"in_features": 512, "out_features": 1,
                                               "scale": 64, "margin": 0.4}),
        "age":   ("AgeLoss_DLDLV2", {}),
        "biattr": ("CEL_Sigmoid", {}),
        "affect": ("MyCrossEntropyLoss", {}),
        "parsing": ("ParsingLoss", {}),
        "align": ("AlignLoss", {"input_size": 512, "heatmap_size": 128,
                                 "heatmap_radius": 5.0}),
    }
    if task_type not in mapping:
        raise ValueError(f"Unknown task type: {task_type}")
    loss_type, kwargs = mapping[task_type]
    return edict({"type": loss_type, "kwargs": edict(kwargs)})


def patch_backbone_model_path(cfg: edict) -> None:
    """Inject the FaRL pretrain path from config into the backbone kwargs."""
    cfg.model.backbone.kwargs.model_path = cfg.paths.farl_pretrain


# ============================================================================
# Model wrapper
# ============================================================================

class SingleImageMultiTaskModel:
    """Load a pretrained Moeface checkpoint and expose per-task inference."""

    def __init__(self, cfg: edict, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        patch_backbone_model_path(cfg)
        task_cfgs = build_task_cfgs_for_aio(cfg)

        backbone = backbone_entry(cfg.model.backbone)
        heads_holder = heads_holder_entry(cfg.model.heads)
        losses_holder = LossesHolder(task_cfgs)
        model = AIOEntry(cfg.model.model_entry, task_cfgs, backbone,
                         heads_holder, losses_holder)

        # Load checkpoint with key filtering
        checkpoint = torch.load(cfg.paths.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        clean = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model_state = model.state_dict()
        loaded, skipped = {}, []
        for k, v in clean.items():
            if k in model_state and model_state[k].shape == v.shape:
                loaded[k] = v
            else:
                skipped.append(k)

        msg = model.load_state_dict(loaded, strict=False)
        self._print_loading_summary(loaded, skipped, msg)

        model.to(device)
        model.eval()
        model.set_mode_to_evaluate()
        self.model = model

    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer_task(self, image_tensor: torch.Tensor, task_name: str) -> Any:
        """Run inference for a single task on a single image tensor."""
        self.model.set_evaluation_task(task_name)
        out = self.model({"image": image_tensor})
        return out["head_output"]

    # ------------------------------------------------------------------

    @staticmethod
    def _print_loading_summary(loaded, skipped, msg) -> None:
        sep = "=" * 80
        print(sep)
        print("Checkpoint loading summary")
        print(f"  loaded keys    : {len(loaded)}")
        print(f"  skipped keys   : {len(skipped)}")
        print(f"  missing keys   : {len(msg.missing_keys)}")
        print(f"  unexpected keys: {len(msg.unexpected_keys)}")
        if skipped:
            print(f"  first 20 skipped: {skipped[:20]}")
        if msg.missing_keys:
            print(f"  first 20 missing: {msg.missing_keys[:20]}")
        if msg.unexpected_keys:
            print(f"  first 20 unexpected: {msg.unexpected_keys[:20]}")
        print(sep)
