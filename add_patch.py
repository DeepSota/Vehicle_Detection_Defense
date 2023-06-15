# -*- coding: utf-8 -*-
import random

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.load_patch_data import PatchTransformer, PatchApplier
from vector2image import showframe_save, showpatch

patchfile = "Patches/patch/yolov5_2.png"

def add_patches(img, label, p):
   p_use = random.random()

   if p_use >(1-p):

        p_loc = random.random()
        p_rotate = random.random()
        rand_loc = True
        do_rotate = False
        if p_loc > 0.5:
             rand_loc = False

        if p_rotate> 0.5:
             do_rotate = True

        patch_size = 300
        mode = torch.randint(0, 3, (1, 1))[0]

        if mode == 0:
             patchfile = "Patches/patch/yolov5_2.png"
             patch_img = Image.open(patchfile).convert('RGB')
             patch_img = patch_img.resize((patch_size, patch_size))
             adv_patch = transforms.ToTensor()(patch_img).cuda()

        elif mode == 1:
             adv_patch = torch.zeros(3, patch_size, patch_size).cuda()
        else:
             adv_patch = torch.randn(3, patch_size, patch_size).cuda()*0.5

        label = torch.from_numpy(label).float()
        image_clean_ref = img
        image_p, label = image_clean_ref, label
        image_tens = transforms.ToTensor()(image_p)
        img_fake_batch = torch.unsqueeze(image_tens, 0).cuda()
        lab_fake_batch = torch.unsqueeze(label, 0).cuda()
        square_size = img.shape[0]
        adv_batch_t = PatchTransformer()(adv_patch, lab_fake_batch, square_size, do_rotate= do_rotate, rand_loc=rand_loc)
        p_img_batch = PatchApplier()(img_fake_batch, adv_batch_t)
        p_img_batch = p_img_batch.squeeze().cpu().numpy().transpose(1, 2, 0)

   else:
        p_img_batch = img

   return p_img_batch

