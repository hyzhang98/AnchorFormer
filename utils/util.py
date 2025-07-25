import numpy as np
import torch
from torch.nn import functional as F

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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def can_softmax(distance, k_sparse):
    bz, heads, n, n_anchors = distance.shape
    if n_anchors <= k_sparse:
        k = n_anchors - 1
    else:
        k = k_sparse
    z = distance
    element, _ = z.sort(dim=-1)
    top_k = element[:, :, :, k].unsqueeze(-1)
    top_k = top_k.repeat(1, 1, 1, n_anchors) + 10 ** -10
    sum_top_k = torch.sum(element[:, :, :, 0:k], dim=-1).unsqueeze(-1)
    sum_top_k = sum_top_k.repeat(1, 1, 1, n_anchors)
    T = top_k - z
    # T = F.relu(T)
    # T = F.tanh(T)
    attention = torch.div(T, k_sparse * top_k - sum_top_k + 1e-7)
    attention = F.relu(attention)
    return attention


