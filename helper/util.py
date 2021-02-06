from __future__ import print_function

import torch
import numpy as np
import torch.distributed as dist
from collections import Counter


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_list(output_list: list, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    batch_size = target.size(0)
    output_avg = torch.mean(torch.stack(output_list), dim=0)
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output_avg.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k == 1:
                top1_list = []
                for output in output_list:
                    maxk = max(topk)
                    _, pred = output.topk(maxk, 1, True, True)
                    pred = pred.t()
                    top1_list.append(pred[0])
                top1_array = np.array([top1.cpu().numpy()
                                       for top1 in top1_list]).transpose()
                top1 = [Counter(top1_array[i]).most_common(1)[0][0]
                        for i in range(batch_size)]
                top1 = torch.Tensor(top1).long().view(-1, batch_size)
                if torch.cuda.is_available():
                    top1 = top1.cuda()
                correct_top1 = top1.eq(target.view(1, -1).expand_as(top1))
                correct_k = correct_top1[:1].view(
                    -1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                correct_k = correct[:k].reshape(-1).float().sum(0,
                                                                keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    torch.manual_seed(0)

    t1 = torch.randn(2, 10)
    t2 = torch.randn(2, 10)
    a = torch.Tensor([7, 7]).long()
    print("t", t1.shape)
    print("a", a.shape)
    res = accuracy_list([t1, t1, t2], a, topk=(1, 5))
    print(res)
    pass
