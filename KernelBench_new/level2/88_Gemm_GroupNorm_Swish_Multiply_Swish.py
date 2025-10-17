import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a GEMM, GroupNorm, Swish, Multiply, and Swish operations.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape)) 
        # Deterministic initialization for benchmarking
        with torch.no_grad():
            torch.manual_seed(42)
            for name in ['gemm', 'group_norm', 'multiply_weight']:
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

    def forward(self, x):
        # (batch_size, in_features) -> (batch_size, out_features)
        x = self.gemm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = self.group_norm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * torch.sigmoid(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * self.multiply_weight
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * torch.sigmoid(x)
        return x

batch_size = 128
in_features = 512
out_features = 1024
num_groups = 16
multiply_weight_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]