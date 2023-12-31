#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-03-27 11:00:10
@LastEditTime: 2019-04-15 09:23:30
"""
import numpy as np
import torch
from torch.autograd import Variable

from EvalBox.Attack.AdvAttack.attack import Attack


class BIM(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: Basic Iterative Method (BIM)
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(BIM, self).__init__(model, device, IsTargeted)

        self.criterion = torch.nn.CrossEntropyLoss()

        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            epsilon:
            eps_iter:
            num_steps:
        } 
        @return: None
        """
        # xs 的偏移值
        self.eps = float(kwargs.get("epsilon", 0.06))
        # 步长的系数
        self.eps_iter = float(kwargs.get("eps_iter", 0.1))
        # 多次迭代的次数
        self.num_steps = int(kwargs.get("num_steps", 15))

    def generate(self, xs=None, ys=None, compute_loss=None):
        """
        @description: 
        @param {
            xs:
            ys:
            device:
        } 
        @return: adv_xs
        """
        device = self.device
        targeted = self.IsTargeted

        copy_xs = np.copy(xs.numpy())
        xs_min, xs_max = copy_xs - self.eps, copy_xs + self.eps

        for _ in range(self.num_steps):
            var_xs = torch.tensor(
                copy_xs, dtype=torch.float, device=device, requires_grad=True
            )
            var_ys = torch.tensor(ys, device=device)

            outputs = self.model(var_xs)
            if targeted:
                loss = -self.criterion(outputs, var_ys)
            else:

                loss, loss_items = compute_loss(outputs[1], var_ys)
            loss.backward()

            grad_sign = var_xs.grad.data.sign().cpu().numpy()
            copy_xs = copy_xs + self.eps_iter * grad_sign
            copy_xs = np.clip(copy_xs, xs_min, xs_max)
            copy_xs = np.clip(copy_xs, 0.0, 1.0)

        adv_xs = torch.from_numpy(copy_xs)

        return adv_xs
