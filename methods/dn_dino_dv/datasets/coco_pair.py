# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Paired RGB+IR COCO dataset for the B3b naive-fusion experiment.

Loads the RGB image from img_folder and the paired IR image (identical
file_name, pixel-aligned pair) from ir_img_folder, then applies the standard
multi-scale training transforms to BOTH modalities with the SAME random draws
(python/numpy/torch RNG state is snapshotted and replayed), so geometric
augmentations (hflip / resize / crop) stay pixel-aligned across modalities.
No separate random crop/flip per modality. Color augmentation: none is used
anywhere in this pipeline (strong_aug is off), so nothing is RGB-only.

Supervision comes from the single annotation json passed via ann_file
(B3: the IR-side canonical GT); the IR image itself is never annotated
separately -- only one detection loss is computed.

The returned image tensor has 6 channels: [rgb(0:3) | ir(3:6)], each
ToTensor+ImageNet-normalized like the single-modality runs.
"""
import copy
import os
import random

import numpy as np
import torch
from PIL import Image

from .coco import CocoDetection


class CocoDetectionPair(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks,
                 aux_target_hacks=None, ir_img_folder=None):
        super().__init__(img_folder, ann_file, transforms, return_masks,
                         aux_target_hacks=aux_target_hacks)
        assert ir_img_folder is not None
        self.ir_root = ir_img_folder

    def __getitem__(self, idx):
        # load rgb image + raw annotations via the torchvision grandparent
        # (NOT CocoDetection.__getitem__, which would already apply transforms)
        import torchvision.datasets
        img, target = torchvision.datasets.CocoDetection.__getitem__(self, idx)
        image_id = self.ids[idx]
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        img_ir = Image.open(os.path.join(self.ir_root, file_name)).convert('RGB')

        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            target_ir = copy.deepcopy(target)
            state = (random.getstate(), np.random.get_state(),
                     torch.get_rng_state())
            img, target = self._transforms(img, target)
            random.setstate(state[0])
            np.random.set_state(state[1])
            torch.set_rng_state(state[2])
            img_ir, target_ir = self._transforms(img_ir, target_ir)
            # same RNG draws must transform the shared boxes identically
            assert torch.allclose(target['boxes'], target_ir['boxes']), \
                f'paired transform diverged on {file_name}'
            assert img.shape[-2:] == img_ir.shape[-2:]

        img = torch.cat([img, img_ir], dim=0)
        return img, target
