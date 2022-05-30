
"""Utilities for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict, OrderedDict
import datetime

from lib.configs.config import cfg
from lib.utils.logging import log_stats
from lib.utils.logging import SmoothedValue
from lib.utils.timer import Timer



class TrainingStats(object):
    """Track vital training statistics."""
    def __init__(self, args, log_period=20, tensorboard_logger=None):
        # Output logging period in SGD iterations
        self.args = args
        self.log_period = log_period
        self.tblogger = tensorboard_logger
        self.tb_ignored_keys = ['iter', 'eta', 'epoch', 'time']
        self.iter_timer = Timer()
        # Window size for smoothing tracked values (with median filtering)
        self.filter_size = log_period
        def create_smoothed_value():
            return SmoothedValue(self.filter_size)
        self.smoothed_losses = defaultdict(create_smoothed_value)
        self.smoothed_metrics = defaultdict(create_smoothed_value)
        self.smoothed_total_loss = SmoothedValue(self.filter_size)


    def IterTic(self):
        self.iter_timer.tic()

    def IterToc(self):
        return self.iter_timer.toc(average=False)

    def ResetIterTimer(self):
        self.iter_timer.reset()

    def UpdateIterStats(self, loss):
        """Update tracked iteration statistics."""
        total_loss = 0
        for k in loss:
            # all losses except the total loss: loss['all']
            if k != 'total_loss':
                self.smoothed_losses[k].AddValue(float(loss[k]))

        total_loss += loss['total_loss']
        self.smoothed_total_loss.AddValue(float(total_loss))

    def LogIterStats(self, cur_iter, cur_epoch, optimizer, val_err={}):
        """Log the tracked statistics."""
        if (cur_iter % self.log_period == 0):
            stats = self.GetStats(cur_iter, cur_epoch, optimizer, val_err)
            log_stats(stats, self.args)
            if self.tblogger:
                self.tb_log_stats(stats, cur_iter)

    def tb_log_stats(self, stats, cur_iter):
        """Log the tracked statistics to tensorboard"""
        for k in stats:
            if k not in self.tb_ignored_keys:
                v = stats[k]
                if isinstance(v, dict):
                    self.tb_log_stats(v, cur_iter)
                else:
                    self.tblogger.add_scalar(k, v, cur_iter)


    def GetStats(self, cur_iter, cur_epoch, optimizer, val_err = {}):
        eta_seconds = self.iter_timer.average_time * (
                cfg.TRAIN.MAX_ITER - cur_iter
        )
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        stats = OrderedDict(
            iter=cur_iter,  # 1-indexed
            time=self.iter_timer.average_time,
            eta=eta,
            total_loss=self.smoothed_total_loss.GetMedianValue(),
            epoch=cur_epoch,
        )
        optimizer_state_dict = optimizer.state_dict()
        lr = {}
        for i in range(len(optimizer_state_dict['param_groups'])):
            lr_name = 'group%d_lr' % i
            lr[lr_name] = optimizer_state_dict['param_groups'][i]['lr']

        stats['lr'] = OrderedDict(lr)
        for k, v in self.smoothed_losses.items():
            stats[k] = OrderedDict([(k, v.GetMedianValue())])

        stats['val_err'] = OrderedDict(val_err)
        return stats
