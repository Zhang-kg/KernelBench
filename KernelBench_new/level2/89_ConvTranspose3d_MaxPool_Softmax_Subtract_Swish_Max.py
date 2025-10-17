import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a sequence of operations:
        - ConvTranspose3d
        - MaxPool3d
        - Softmax
        - Subtract
        - Swish
        - Max
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels)) # Assuming subtraction is element-wise across channels
        # Deterministic initialization for benchmarking
        with torch.no_grad():
            torch.manual_seed(42)
            for name in ['conv_transpose', 'max_pool', 'subtract']:
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
        x = self.max_pool(x)
        x = torch.softmax(x, dim=1) # Apply softmax across channels (dim=1)
        x = x - self.subtract.view(1, -1, 1, 1, 1) # Subtract across channels
        x = torch.sigmoid(x) * x # Swish activation
        x = torch.max(x, dim=1)[0] # Max pooling across channels
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding]