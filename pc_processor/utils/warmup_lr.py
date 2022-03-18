# This file is covered by the LICENSE file in the root of this project.

import torch.optim.lr_scheduler as toptim


class WarmupLR(toptim._LRScheduler):
    """ Warmup learning rate scheduler.
        Initially, increases the learning rate from 0 to the final value, in a
        certain number of steps. After this number of steps, each step decreases
        LR exponentially.
    """

    def __init__(self, optimizer, lr, warmup_steps, momentum, decay):
        # cyclic params
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.momentum = momentum
        self.decay = decay

        # cap to one
        self.warmup_steps = max(self.warmup_steps, 1)
        # cyclic lr
        self.initial_scheduler = toptim.CyclicLR(self.optimizer,
                                                 base_lr=0,
                                                 max_lr=self.lr,
                                                 step_size_up=self.warmup_steps,
                                                 step_size_down=self.warmup_steps,
                                                 cycle_momentum=False,
                                                 base_momentum=self.momentum,
                                                 max_momentum=self.momentum)

        # our params
        self.last_epoch = -1  # fix for pytorch 1.1 and below
        self.finished = False  # am i done
        super().__init__(optimizer)

    def get_lr(self):
        return [self.lr * (self.decay ** self.last_epoch) for _ in self.base_lrs]

    def step(self, epoch=None):
        if (
            not self.finished
            and self.initial_scheduler.last_epoch < self.warmup_steps
        ):
            return self.initial_scheduler.step(epoch)
        if not self.finished:
            self.base_lrs = [self.lr for _ in self.base_lrs]
            self.finished = True
        return super(WarmupLR, self).step(epoch)



class WarmupCosineLR(toptim._LRScheduler):
    """ Warmup learning rate scheduler.
        Initially, increases the learning rate from 0 to the final value, in a
        certain number of steps. After this number of steps, each step decreases
        LR exponentially.
    """

    def __init__(self, optimizer, lr, warmup_steps, momentum, max_steps):
        # cyclic params
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.momentum = momentum

        # cap to one
        self.warmup_steps = max(self.warmup_steps, 1)
        # cyclic lr
        self.cosine_scheduler = toptim.CosineAnnealingLR(
            self.optimizer, T_max=max_steps)

        self.initial_scheduler = toptim.CyclicLR(self.optimizer,
                                                 base_lr=0,
                                                 max_lr=self.lr,
                                                 step_size_up=self.warmup_steps,
                                                 step_size_down=self.warmup_steps,
                                                 cycle_momentum=False,
                                                 base_momentum=self.momentum,
                                                 max_momentum=self.momentum)

        # our params
        self.last_epoch = -1  # fix for pytorch 1.1 and below
        self.finished = False  # am i done
        super().__init__(optimizer)
        
    def step(self, epoch=None):
        if self.finished:
            return self.cosine_scheduler.step(epoch)
        elif self.initial_scheduler.last_epoch >= self.warmup_steps:
            self.base_lrs = [self.lr for _ in self.base_lrs]
            self.finished = True
            return self.cosine_scheduler.step(epoch)
        else:
            return self.initial_scheduler.step(epoch)

