import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        # Deterministic initialization for benchmarking
        with torch.no_grad():
            torch.manual_seed(42)
            for name in ['conv', 'bias', 'scale', 'group_norm']:
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
        x = self.conv(x)
        x = x + self.bias
        x = x * self.scale
        x = torch.sigmoid(x)
        x = self.group_norm(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]