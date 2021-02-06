import torch.nn as nn
import torch


class L2Loss(nn.Module):
    def __init__(self, div_element=False, norm=False):
        super(L2Loss, self).__init__()
        self.div_element = div_element
        self.norm = norm

    def forward(self, output, target):
        loss = torch.Tensor([0]).cuda()
        loss += torch.sum(torch.pow(torch.add(output, -1, target), 2))
        loss = loss / output.size(0)
        if self.div_element:
            loss = loss / output.numel()
        else:
            loss = loss / output.size(0) / 2
        return loss