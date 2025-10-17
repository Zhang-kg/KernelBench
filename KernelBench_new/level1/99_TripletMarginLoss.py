import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(Model, self).__init__()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin)
        # Deterministic initialization for benchmarking
        with torch.no_grad():
            torch.manual_seed(42)
            for name in ['loss_fn']:
                attr = getattr(self, name, None)
                if attr is None:
                    continue
                if hasattr(attr, 'weight'):
                    attr.weight.copy_(torch.randn_like(attr.weight))
                    attr.weight.requires_grad = False
                    bias_attr = getattr(attr, 'bias', None)
                    if bias_attr is not None:
                        bias_attr.zero_()
                        bias_attr.requires_grad = False
                elif isinstance(attr, torch.nn.Parameter):
                    attr.copy_(torch.randn_like(attr))
                    attr.requires_grad = False
                elif isinstance(attr, torch.Tensor):
                    attr.copy_(torch.randn_like(attr))
                    attr.requires_grad = False

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [1.0]  # Default margin
