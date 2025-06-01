# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:05:16 2022

@author: talha
"""

import math
import numpy as np

class LR_Scheduler(object):
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch  # Total iterations
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, current_iter):
        T = current_iter
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(math.pi * T / self.N))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - T / self.N), 0.9)
        elif self.mode == 'step':
            epoch = T // self.iters_per_epoch
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplementedError
        # Warm-up learning rate
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * T / self.warmup_iters
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def cyclical_learning_rate(batch_step,
                           step_size,
                           base_lr=0.001,
                           max_lr=0.006,
                           mode='triangular',
                           gamma=0.999995):

    cycle = np.floor(1 + batch_step / (2. * step_size))
    x = np.abs(batch_step / float(step_size) - 2 * cycle + 1)

    lr_delta = (max_lr - base_lr) * np.maximum(0, (1 - x))
    
    if mode == 'triangular':
        pass  # No modification
    elif mode == 'triangular2':
        lr_delta = lr_delta * (1 / (2. ** (cycle - 1)))
    elif mode == 'exp_range':
        lr_delta = lr_delta * (gamma ** batch_step)
    else:
        raise ValueError('mode must be "triangular", "triangular2", or "exp_range"')
        
    lr = base_lr + lr_delta
    
    return lr

#  convert this to a class
class CyclicLR(object):
    def __init__(self, optimizer, base_lr, max_lr, step_size, mode='triangular', gamma=0.999995):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.batch_step = 0
        self.lr = base_lr

    def step(self):
        self.lr = cyclical_learning_rate(self.batch_step, self.step_size,
                                         self.base_lr, self.max_lr,
                                         self.mode, self.gamma)
        self._adjust_learning_rate(self.lr)
        self.batch_step += 1

    def _adjust_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.lr

    def get_batch_step(self):
        return self.batch_step

    def get_cycle(self):
        return np.floor(1 + self.batch_step / (2. * self.step_size))

    def get_x(self):
        return np.abs(self.batch_step / float(self.step_size) - 2 * self.get_cycle() + 1)

    def get_lr_delta(self):
        return (self.max_lr - self.base_lr) * np.maximum(0, (1 - self.get_x()))

    def get_mode(self):
        return self.mode

    def get_gamma(self):
        return self.gamma

    def get_optimizer(self):
        return self.optimizer

    def get_base_lr(self):
        return self.base_lr

    def get_max_lr(self):
        return self.max_lr

    def get_step_size(self):
        return self.step_size







# class LR_Scheduler(object):
    # """Learning Rate Scheduler

    # Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    # Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    # Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    # Args:
    #     args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
    #       :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
    #       :attr:`args.lr_step`

    #     iters_per_epoch: number of iterations per epoch
    # """
    # def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
    #              lr_step=0, warmup_epochs=0):
    #     self.mode = mode
    #     print('Using {} LR Scheduler!'.format(self.mode))
    #     self.lr = base_lr
    #     if mode == 'step':
    #         assert lr_step
    #     self.lr_step = lr_step
    #     self.iters_per_epoch = iters_per_epoch
    #     self.N = num_epochs * iters_per_epoch
    #     self.epoch = -1
    #     self.warmup_iters = warmup_epochs * iters_per_epoch
        

    # def __call__(self, optimizer, i, epoch):
    #     T = epoch * self.iters_per_epoch + i
    #     if self.mode == 'cos':
    #         lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
    #     elif self.mode == 'poly':
    #         lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
    #     elif self.mode == 'step':
    #         lr = self.lr * (0.1 ** (epoch // self.lr_step))
    #     else:
    #         raise NotImplemented
    #     # warm up lr schedule
    #     if self.warmup_iters > 0 and T < self.warmup_iters:
    #         lr = lr * 1.0 * T / self.warmup_iters
    #     if epoch > self.epoch:
    #         # print(f'=> Epoches: {epoch}, Learning Rate: {lr:.7f}')
    #         self.epoch = epoch
    #     assert lr >= 0
    #     self._adjust_learning_rate(optimizer, lr)

    # def _adjust_learning_rate(self, optimizer, lr):
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #         # print(f"Updated learning rate for param group to {lr:.7f}")


#     # THIS WAS NOT WORKING with the MAE model
#     # def _adjust_learning_rate(self, optimizer, lr):
#     #     if len(optimizer.param_groups) == 1:
#     #         optimizer.param_groups[0]['lr'] = lr
#     #     else:
#     #         # enlarge the lr at the head
#     #         for i in range(len(optimizer.param_groups)):
#     #             if optimizer.param_groups[i]['lr'] > 0: optimizer.param_groups[i]['lr'] = lr
#             # optimizer.param_groups[0]['lr'] = lr
#             # for i in range(1, len(optimizer.param_groups)):
#             #     optimizer.param_groups[i]['lr'] = lr * 10