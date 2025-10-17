import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        # Deterministic initialization for benchmarking
        with torch.no_grad():
            torch.manual_seed(42)
            for name in ['conv_transpose', 'relu', 'group_norm']:
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
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        x = self.relu(x)
        x = self.group_norm(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 8, 16, 16
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]