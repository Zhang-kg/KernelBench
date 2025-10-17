import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling, 
    global average pooling, and clamps the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1
        # Deterministic initialization for benchmarking
        with torch.no_grad():
            torch.manual_seed(42)
            for name in ['conv_transpose', 'maxpool', 'global_avg_pool']:
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
        x = self.conv_transpose(x)
        x = x * self.scale
        x = self.maxpool(x)
        x = self.global_avg_pool(x)
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size]